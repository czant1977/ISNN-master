# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
from torch import nn
import torch
import math
import random


class ISNN(nn.Module):
    def __init__(self, num_classes, dataset, seg, args, bias=True):
        super(ISNN, self).__init__()

        self.dim1 = 256
        self.dataset = dataset
        self.seg = seg
        num_joint = 25
        bs = args.batch_size
        if args.train:
            self.spa = self.one_hot(bs, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(bs, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()

        else:
            self.spa = self.one_hot(32 * 5, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(32 * 5, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()

        self.tem_embed = embed(self.seg, 64 * 8, norm=False, bias=bias)
        self.spa_embed = embed(num_joint, 64, norm=False, bias=bias)
        self.spa_embed1 = embed(6, 25, norm=False, bias=bias)
        self.joint_embed = embed(6, 64, norm=True, bias=bias)
        self.col_embed = embed(25, 64, norm=True, bias=bias)
        self.dis_embed = embed(3, 64, norm=True, bias=bias)
        self.cos_embed = embed(25, 64, norm=True, bias=bias)

        self.dif_embed = embed(6, 64, norm=True, bias=bias)
        self.joint_embed1 = embed(6, 64, norm=True, bias=bias)
        self.joint_embed2 = embed(3, 64, norm=True, bias=bias)
        self.joint_embed3 = embed(3, 64, norm=True, bias=bias)
        self.dif_embed1 = embed(3, 64, norm=True, bias=bias)
        self.dis_embed1 = embed(3, 64, norm=True, bias=bias)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.cnn = local(seg, self.dim1 * 2, self.dim1 * 4, bias=bias)
        self.cnn_1 = local(seg, self.dim1 * 2, self.dim1 * 4, bias=bias)
        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.compute_g2 = compute_g_spa_1(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.dim1 * 2, bias=bias)
        self.gcn4 = gcn_spa(self.dim1 * 2, self.dim1 * 2, bias=bias)
        self.gcn1_1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn2_1 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3_1 = gcn_spa(self.dim1, self.dim1 * 2, bias=bias)
        self.gcn4_1 = gcn_spa(self.dim1 * 2, self.dim1 * 2, bias=bias)
        self.fc = nn.Linear(self.dim1 * 4, num_classes)
        self.fc_1 = nn.Linear(self.dim1 * 4, num_classes)
        self.integration_param1 = nn.Parameter(
            torch.FloatTensor([0.5]), requires_grad=True)

        self.res_cnn = nn.Conv2d(self.dim1 // 2, self.dim1 * 2, kernel_size=1, bias=bias)
        self.res_cnn_1 = nn.Conv2d(self.dim1 // 2, self.dim1 * 2, kernel_size=1, bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)
        nn.init.constant_(self.gcn4.w.cnn.weight, 0)
        nn.init.constant_(self.gcn1_1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2_1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3_1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn4_1.w.cnn.weight, 0)

    def forward(self, input):

        # joint stream
        input1 = input[:, :, 0:75]
        input2 = input[:, :, 75:150]
        bs, step, dim = input1.size()
        num_joints = dim // 3
        input1 = input1.view(bs, step, num_joints, 3)
        input2 = input2.view(bs, step, num_joints, 3)
        input = torch.cat((input1, input2), 3)
        # bone stream
        bone1 = self.compute_bone(input1)
        bone2 = self.compute_bone(input2)
        bone = torch.cat((bone1, bone2), dim=3)
        # Dynamic Representation-joint stream
        input1 = self.compute_input1(input1)
        input2 = self.compute_input1(input2)
        input, spa1 = self.compute_input(input)
        tem1 = self.tem_embed(self.tem)
        aerfa = self.scaledTanh(self.integration_param1)
        # Dynamic Representation-bone stream
        input_bone = bone.permute(0, 3, 2, 1)
        input_bone = self.joint_embed1(input_bone)
        input_bone = torch.cat((input_bone, spa1), dim=1)
        bone1 = self.compute_bone_1(bone1)
        bone2 = self.compute_bone_1(bone2)
        # compute MAM
        MAM_J = self.compute_g2(input1, input2)
        MAM_B = self.compute_g1(bone1, bone2)
        # Joint Module-joint stream
        x = self.res_cnn(input)
        input = self.gcn1(input, MAM_J)
        input = self.gcn2(input, MAM_J)
        input = self.gcn3(input, MAM_J)
        input = self.gcn4(input, MAM_J)
        # Frame Module-joint stream
        input = input + tem1 + x
        input = self.cnn(input)
        # Classification-joint stream
        output = self.maxpool(input)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        # Joint Module-bone stream
        x1 = self.res_cnn_1(input_bone)
        input1 = self.gcn1_1(input_bone, MAM_B)
        input1 = self.gcn2_1(input1, MAM_B)
        input1 = self.gcn3_1(input1, MAM_B)
        input1 = self.gcn4_1(input1, MAM_B)
        # Frame Module-bone stream
        input1 = input1 + tem1 + x1
        input1 = self.cnn_1(input1)
        # Classification-bone stream
        output1 = self.maxpool(input1)
        output1 = torch.flatten(output1, 1)
        output1 = self.fc_1(output1)
        # Classification Weighted fusion
        output = aerfa * output + (1 - aerfa) * output1

        return output

    def scaledTanh(self, param):

        return (torch.tanh(param) + 1.0) / 2.0

    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)
        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot

    def compute_bone(self, a):
        edge = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                (22, 23), (23, 8), (24, 25), (25, 12)]
        for group in edge:
            i = group[0]
            j = group[1]
            x = a[:, :, i - 1:i, :].squeeze()
            y = a[:, :, j - 1:j, :].squeeze()
            dis = (y - x).unsqueeze(2)
            if i == 1:
                l = dis
            elif i == 20:
                l = torch.cat((l, dis), dim=2)
                temp = a[:, :, i:i + 1, :]
                l = torch.cat((l, temp), dim=2)
            else:
                l = torch.cat((l, dis), dim=2)
        return l

    def compute_bone_1(self, bone):
        bone = bone.permute(0, 3, 2, 1).contiguous()
        bone = self.joint_embed3(bone)
        spa1 = self.spa_embed(self.spa)
        bone = torch.cat([bone, spa1], 1)
        return bone

    def compute_input(self, input):

        input = input.permute(0, 3, 2, 1).contiguous()
        dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
        dif = torch.cat([dif.new(input.size(0), dif.size(1), input.size(2), 1).zero_(), dif], dim=-1)
        pos = self.joint_embed(input)
        spa1 = self.spa_embed(self.spa)
        dif = self.dif_embed(dif)
        dy = pos + dif
        input = torch.cat([dy, spa1], 1)
        return input, spa1

    def compute_input1(self, input):

        input = input.permute(0, 3, 2, 1).contiguous()
        dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
        dif = torch.cat([dif.new(input.size(0), dif.size(1), input.size(2), 1).zero_(), dif], dim=-1)
        pos = self.joint_embed2(input)
        spa1 = self.spa_embed(self.spa)
        dif = self.dif_embed1(dif)
        dy = pos + dif
        input = torch.cat([dy, spa1], 1)
        return input


class norm_data(nn.Module):
    def __init__(self, dim=64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim * 25)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.contiguous().view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x


class embed(nn.Module):
    def __init__(self, dim=3, dim1=128, norm=True, bias=False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x


class cnn1x1(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x


class local(nn.Module):
    def __init__(self, seg, dim1=3, dim2=3, bias=False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, seg))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        x1 = self.maxpool(x1)
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias=False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)

    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        # 将张量的维度换位
        x = g.matmul(x)
        # 返回两个数组的矩阵乘积
        x = x.permute(0, 3, 2, 1).contiguous()
        # 将张量的维度换位
        x = self.w(x) + self.w1(x1)
        # 计算矩阵的和
        x = self.relu(self.bn(x))
        # Relu激活
        return x


class compute_g_spa(nn.Module):
    def __init__(self, dim1=64 * 3, dim2=64 * 3, bias=False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x2).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g


class compute_g_spa_1(nn.Module):
    def __init__(self, dim1=64 * 3, dim2=64 * 3, bias=False):
        super(compute_g_spa_1, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x2).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g
