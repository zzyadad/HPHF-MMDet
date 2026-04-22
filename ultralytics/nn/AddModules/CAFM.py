import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange


class CAFM(nn.Module):  # Cross Attention Fusion Module
    def __init__(self, channels):
        super(CAFM, self).__init__()

        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1)

        self.avg1 = nn.Conv2d(channels, 64, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(channels, 64, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(channels, 64, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(channels, 64, 1, stride=1, padding=0)

        self.avg11 = nn.Conv2d(64, channels, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(64, channels, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(64, channels, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(64, channels, 1, stride=1, padding=0)

    def forward(self, x):
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        f1 = rgb_fea.reshape([bs, c, -1])
        f2 = ir_fea.reshape([bs, c, -1])

        # 计算第一个特征的平均和最大池化特征
        avg_1 = torch.mean(f1, dim=-1, keepdim=True).unsqueeze(-1)
        max_1, _ = torch.max(f1, dim=-1, keepdim=True)
        max_1 = max_1.unsqueeze(-1)

        avg_1 = F.relu(self.avg1(avg_1))
        max_1 = F.relu(self.max1(max_1))
        avg_1 = self.avg11(avg_1).squeeze(-1)
        max_1 = self.max11(max_1).squeeze(-1)
        a1 = avg_1 + max_1

        # 计算第二个特征的平均和最大池化特征
        avg_2 = torch.mean(f2, dim=-1, keepdim=True).unsqueeze(-1)
        max_2, _ = torch.max(f2, dim=-1, keepdim=True)
        max_2 = max_2.unsqueeze(-1)

        avg_2 = F.relu(self.avg2(avg_2))
        max_2 = F.relu(self.max2(max_2))
        avg_2 = self.avg22(avg_2).squeeze(-1)
        max_2 = self.max22(max_2).squeeze(-1)
        a2 = avg_2 + max_2

        # 计算交叉注意力
        cross = torch.matmul(a1, a2.transpose(1, 2))

        # 根据交叉注意力调整特征
        a1_att = torch.matmul(F.softmax(cross, dim=-1), f1)
        a2_att = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2)

        # 恢复特征的原始形状
        a1_att = a1_att.reshape([bs, c, h, w])
        a2_att = a2_att.reshape([bs, c, h, w])

        # 计算空间注意力
        avg_out_1 = torch.mean(a1_att, dim=1, keepdim=True)
        max_out_1, _ = torch.max(a1_att, dim=1, keepdim=True)
        a1_spatial = torch.cat([avg_out_1, max_out_1], dim=1)
        a1_spatial = F.relu(self.conv1_spatial(a1_spatial))
        a1_spatial = self.conv2_spatial(a1_spatial)
        a1_spatial = a1_spatial.reshape([bs, 1, -1])
        a1_spatial = F.softmax(a1_spatial, dim=-1)

        avg_out_2 = torch.mean(a2_att, dim=1, keepdim=True)
        max_out_2, _ = torch.max(a2_att, dim=1, keepdim=True)
        a2_spatial = torch.cat([avg_out_2, max_out_2], dim=1)
        a2_spatial = F.relu(self.conv1_spatial(a2_spatial))
        a2_spatial = self.conv2_spatial(a2_spatial)
        a2_spatial = a2_spatial.reshape([bs, 1, -1])
        a2_spatial = F.softmax(a2_spatial, dim=-1)

        # 应用注意力权重到特征上
        f1_att = f1 * a1_spatial + f1
        f2_att = f2 * a2_spatial + f2

        # 调整特征维度，恢复到 (B, C, H, W) 形状
        f1_out = f1_att.view(bs, c, h, w)
        f2_out = f2_att.view(bs, c, h, w)

        return f1_out, f2_out
    