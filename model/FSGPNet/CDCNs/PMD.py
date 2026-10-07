import torch
import torch.nn as nn
import torch.nn.functional as F

class PMDModule(nn.Module):
    """
    PMD 模块：
      - 对每个输入通道采用组卷积分别计算一阶和二阶导数。
      - 二阶导数算子 (conv_xx, conv_yy) 改为 5×5。
      - 计算公式：
            PMD = (u_x^2 * u_yy - 2*u_x*u_y*u_xy + u_y^2 * u_xx) / (2*(u_x^2+u_y^2+eps))
      - 加入梯度幅值正则项（lambda_param）以增强边缘信息。
      - 若 in_channels != out_channels，则用 1×1 卷积映射通道数。
    """
    def __init__(self, in_channels, out_channels, lambda_param=0.1):
        super(PMDModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lambda_param = nn.Parameter(torch.tensor(lambda_param, dtype=torch.float32))

        ##############################
        # 一阶与混合导数算子 (仍为3×3)
        ##############################
        self.conv_x = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            bias=False, groups=in_channels
        )
        self.conv_y = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            bias=False, groups=in_channels
        )
        self.conv_xy = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            bias=False, groups=in_channels
        )

        # 初始化为不可训练的差分算子
        kernel_x = torch.tensor([[[[0, 0, 0],
                                   [-1, 0, 1],
                                   [0, 0, 0]]]], dtype=torch.float32)
        self.conv_x.weight = nn.Parameter(kernel_x.repeat(in_channels, 1, 1, 1), requires_grad=False)

        kernel_y = torch.tensor([[[[0, -1, 0],
                                   [0,  0, 0],
                                   [0,  1, 0]]]], dtype=torch.float32)
        self.conv_y.weight = nn.Parameter(kernel_y.repeat(in_channels, 1, 1, 1), requires_grad=False)

        kernel_xy = torch.tensor([[[[1, 0, -1],
                                    [0, 0,  0],
                                    [-1, 0, 1]]]], dtype=torch.float32)
        self.conv_xy.weight = nn.Parameter(kernel_xy.repeat(in_channels, 1, 1, 1), requires_grad=False)

        ##############################
        # 二阶导数算子 (改为5×5)
        ##############################
        self.conv_xx = nn.Conv2d(
            in_channels, in_channels, kernel_size=5, padding=2,
            bias=False, groups=in_channels
        )
        self.conv_yy = nn.Conv2d(
            in_channels, in_channels, kernel_size=5, padding=2,
            bias=False, groups=in_channels
        )

        # 下面是示例的二阶算子 (W_xx, W_yy)
        # 你也可以根据需求定义其他形式的 5×5 核
        kernel_xx_5x5 = torch.tensor([[[[0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [1, 0, -2, 0, 1],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]]]], dtype=torch.float32)
        kernel_yy_5x5 = torch.tensor([[[[0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, -2, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0]]]], dtype=torch.float32)

        self.conv_xx.weight = nn.Parameter(kernel_xx_5x5.repeat(in_channels, 1, 1, 1), requires_grad=False)
        self.conv_yy.weight = nn.Parameter(kernel_yy_5x5.repeat(in_channels, 1, 1, 1), requires_grad=False)

        # 若输入通道 != 输出通道，则使用 1×1 卷积做映射
        if in_channels != out_channels:
            self.conv_expand = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_expand = None

    def forward(self, x):
        # x: (B, in_channels, H, W)
        with torch.no_grad():
            ux = self.conv_x(x)      # 一阶 x
            uy = self.conv_y(x)      # 一阶 y
            uxy = self.conv_xy(x)    # 混合导数
            uxx = self.conv_xx(x)    # 二阶 x (5×5)
            uyy = self.conv_yy(x)    # 二阶 y (5×5)

        eps = 1e-6
        numerator = ux**2 * uyy - 2*ux*uy*uxy + uy**2 * uxx
        denominator = 2*(ux**2 + uy**2 + eps)
        pmd = numerator / denominator

        # 梯度幅值正则项
        grad_mag = torch.sqrt(ux**2 + uy**2 + eps)
        pmd = pmd + self.lambda_param * grad_mag

        if self.conv_expand is not None:
            pmd = self.conv_expand(pmd)
        return pmd


###############################################
# 修改后的 PMDEncoderBlock：去除残差连接，直接输出处理后的特征
###############################################
class PMDBlock(nn.Module):
    """
    Unet Encoder 部分的 PMD 模块块：
      - 首先使用 PMDModule 对输入进行形状感知处理，
      - 然后经过两层卷积融合特征，
      - 此处不再采用残差连接，直接输出融合后的特征。

    输入尺寸：(B, in_channels, H, W)
    输出尺寸：(B, out_channels, H, W)
    """

    def __init__(self, in_channels, out_channels):
        super(PMDBlock, self).__init__()
        # PMD模块：将 in_channels 映射到 out_channels（如二者不一致）
        self.pmd = PMDModule(in_channels, in_channels, lambda_param=0.1)
        # 后续卷积融合层
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # 先通过 PMD 模块处理，获得形状感知特征（通道数为 out_channels）
        pmd_out = self.pmd(x)
        fused = self.conv(pmd_out)
        # 不再进行残差融合，直接输出融合后的特征
        return fused


###############################################
# 示例：使用 PMDEncoderBlock
###############################################
if __name__ == '__main__':
    # 假设 Unet encoder 某层输入尺寸为 (B, 64, 128, 128)，目标输出尺寸为 (B, 128, 128, 128)
    x = torch.randn(1, 64, 128, 128)
    encoder_block = PMDBlock(in_channels=64, out_channels=128)
    out = encoder_block(x)
    print("Encoder block output shape:", out.shape)
