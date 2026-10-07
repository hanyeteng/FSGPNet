# ------------------------------------------------------------------#
# Code Structure of HS-FPN (https://arxiv.org/abs/2412.10116)
# HS-FPN
# ├── HFP (High Frequency Perception Module)
# │   ├── DctSpatialInteraction (Spatial Path of HFP)
# │   └── DctChannelInteraction (Channel Path of HFP)
# └── SDP&SDP_Large (Spatial Dependency Perception Module
# -----------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as DCT
import unittest




# 自定义ConvModule替代mmcv中的实现
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, bias=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )
        self.norm = nn.GroupNorm(32, out_channels) if out_channels % 32 == 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# ------------------------------------------------------------------#
# Spatial Path of HFP
# Only p1&p2 use dct to extract high_frequency response
# ------------------------------------------------------------------#
class DctSpatialInteraction(nn.Module):
    def __init__(self, in_channels, ratio, isdct=True):
        super(DctSpatialInteraction, self).__init__()
        self.ratio = ratio
        self.isdct = isdct

        self.spatial1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        _, _, h0, w0 = x.size()
        if not self.isdct:
            return x.mean(dim=1, keepdim=True)
        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        weight = weight.view(1, h0, w0).expand_as(idct)
        dct = idct * weight
        dct_ = DCT.idct_2d(dct, norm='ortho')
        return dct_.mean(dim=1, keepdim=True)

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


# ------------------------------------------------------------------#
# Channel Path of HFP
# Only p1&p2 use dct to extract high_frequency response
# ------------------------------------------------------------------#
class DctChannelInteraction(nn.Module):
    def __init__(self, in_channels, patch, ratio, isdct=True):
        super(DctChannelInteraction, self).__init__()
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct

        # Channel attention path (for 'out' computation)
        if in_channels >= 32:
            self.channel1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=32, bias=False)
            self.channel2x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=32, bias=False)
        else:
            self.channel1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)
            self.channel2x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)
        self.relu = nn.ReLU()

        # Kernel generator path (for 'dynamic_kernel' computation)
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 3**2, kernel_size=1)
        )

    def compute_out(self, x):
        """Original DctChannelInteraction's output computation"""
        n, c, h, w = x.size()

        if not self.isdct:
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))
            return x * torch.sigmoid(self.channel2x1(channel))

        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, h, w).expand_as(idct)
        dct = idct * weight
        dct_ = DCT.idct_2d(dct, norm='ortho')

        amaxp = F.adaptive_max_pool2d(dct_, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_, output_size=(self.h, self.w))
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return x * torch.sigmoid(self.channel2x1(channel))

    def compute_dynamic_kernel(self, x):
        """Kernel generator computation matching DynamicSpatialAttention's style"""
        return self.kernel_generator(x)  # [B, 9, 1, 1]

    def forward(self, x):
        dynamic_kernel = self.compute_dynamic_kernel(x)
        out = self.compute_out(x)
        return dynamic_kernel, out

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight

    # ------------------------------------------------------------------#


# High Frequency Perception Module HFP
# ------------------------------------------------------------------#
class DynamicHFP(nn.Module):
    def __init__(self, in_channels, ratio, patch=(8, 8), isdct=True):
        super(DynamicHFP, self).__init__()
        self.kernel_size=3
        self.spatial_mean = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct)
        self.kernel_generator = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct)

        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, in_channels) if in_channels % 32 == 0 else nn.Identity()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        B, C, H, W = x.shape

        # 1. 每个样本生成一个动态卷积核 [B, k*k, 1, 1] → [B, 1, k, k]
        # kernels = self.kernel_generator(x).view(B, 1, self.kernel_size, self.kernel_size)

        kernel_out,channel_out = self.kernel_generator(x)
        kernels=kernel_out.view(B, 1, self.kernel_size, self.kernel_size)

        x_mean = self.spatial_mean(x)
        # 3. reshape 成 grouped convolution 所需格式
        x_mean = x_mean.view(1, B, H, W)  # → [1, B, H, W]
        kernels = kernels.view(B, 1, self.kernel_size, self.kernel_size)  # [B, 1, k, k]

        # 4. 执行 grouped convolution，每个 kernel 只作用于对应的样本
        att = F.conv2d(
            x_mean,
            weight=kernels,
            padding=self.kernel_size // 2,
            groups=B
        )
        # 5. reshape 回原格式 + sigmoid
        att = att.view(B, 1, H, W)
        att = self.sigmoid(att)
        # 6. 应用注意力图
        return self.out(x * att+channel_out)

# ------------------------------------------------------------------#

class DynamicSpatialAttention(nn.Module):
    def __init__(self, in_channels=32, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, kernel_size**2, kernel_size=1)  # [B, k*k, 1, 1]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 每个样本生成一个动态卷积核 [B, k*k, 1, 1] → [B, 1, k, k]
        kernels = self.kernel_generator(x).view(B, 1, self.kernel_size, self.kernel_size)
        # 2. 对每个样本取通道平均 [B, 1, H, W]
        x_mean = x.mean(dim=1, keepdim=True)
        # 3. reshape 成 grouped convolution 所需格式
        x_mean = x_mean.view(1, B, H, W)  # → [1, B, H, W]
        kernels = kernels.view(B, 1, self.kernel_size, self.kernel_size)  # [B, 1, k, k]
        # 4. 执行 grouped convolution，每个 kernel 只作用于对应的样本
        att = F.conv2d(
            x_mean,
            weight=kernels,
            padding=self.kernel_size // 2,
            groups=B
        )
        # 5. reshape 回原格式 + sigmoid
        att = att.view(B, 1, H, W)
        att = self.sigmoid(att)
        # 6. 应用注意力图
        return x * att
