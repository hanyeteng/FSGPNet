import torch
import torch.nn as nn
from torch.nn import Flatten
import torch.nn.functional as F
from .APconv import PConv
from .CDCNs.Gradient_model import ExpansionContrastModule
from .CDCNs.PMD import PMDBlock
# from .AttentionModule import *
from .SCTransNet import *
from .SCTransNet import get_CTranS_config
from .context import AGCB_Element
from .hs_fpn9 import DynamicHFP
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()



class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU', kernel_size=3):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding='same')
        # self.conv = PConv(c1=in_channels, c2=out_channels, k=3, s=1)

        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        self.sattn = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1),
                                   nn.Sigmoid())

    # d深层特征，c, f(skip_x), xin skip_x
    def forward(self, d, c, xin):
        d = self.up(d)
        d = self.sattn(xin) * d
        x = torch.cat([c, d], dim=1)
        x = self.nConvs(x)
        return x


        # self.SelfAttn_p4 = HFP(out_channels, ratio=None, isdct=False)
        # self.SelfAttn_p3 = HFP(out_channels, ratio=None, isdct=False)
        # self.SelfAttn_p2 = HFP(out_channels, ratio=ratio, patch=(8, 8), isdct=True)
        # self.SelfAttn_p1 = HFP(out_channels, ratio=ratio, patch=(16, 16), isdct=True)

class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, patch, isdct,stride=1):
        super(Res_block, self).__init__()

        # 主分支（这里使用 PConv，可替换为标准卷积）
        self.conv1 = PConv(c1=in_channels, c2=out_channels, k=3, s=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = PConv(c1=out_channels, c2=out_channels, k=3, s=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # PMD 分支：保持输入通道数不变，对主分支输出进行形状感知处理
        self.pmd = PMDBlock(in_channels, out_channels)

        # 如果输入尺寸或通道不匹配，则用 shortcut 调整（残差连接）
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        self.SelfAttn = DynamicHFP(out_channels, ratio=ratio, patch=patch, isdct=isdct)
        self.conv0= nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.act=nn.Sigmoid()
    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        w_out= self.conv0(x)
        w_out = self.SelfAttn(w_out)
        w_out = self.act(w_out)


        # 计算 PMD 分支：对主分支输出进行边缘与几何信息提取
        pmd_out = self.pmd(x)
        # 融合 PMD 分支（这里采用加和融合方式）
        out = out + pmd_out+ w_out*out

        # 加入残差连接，并激活
        out += residual
        out = self.relu(out)
        return out


class FSGPNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, img_size=256, vis=False, mode='train', deepsuper=True):
        super().__init__()
        self.vis = vis
        self.deepsuper = deepsuper
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes

        block = Res_block

        # skip connection transformer
        config = get_CTranS_config()

        in_channels = config.base_channel  # basic channel 16
        self.mtc = ChannelTransformer(config, vis, img_size,
                                      channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8],
                                      patchSize=config.patch_sizes)

        self.pool = nn.MaxPool2d(2, 2)
        # 1*1卷积升维 1->16
        self.inc = self._make_layer(block, n_channels, in_channels,(0.25,0.25), (32, 32),True,1)
        # 1*1卷积升维 16->32
        self.encoder1 = self._make_layer(block, in_channels, in_channels * 2,(0.25,0.25), (16, 16),True,1)
        # 1*1卷积升维 32->64
        self.encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, (0.25,0.25),(8, 8),True,1)
        # 1*1卷积升维 64->128
        self.encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, None,(8, 8),False,1)
        # 1*1卷积升维 128->128
        self.encoder4 = self._make_layer(block, in_channels * 8, in_channels * 8, None,(8, 8),False,1)
        # self.encoder5 = self._make_layer(block, in_channels * 8, in_channels * 8, 1)
        # self.encoder6 = self._make_layer(block, in_channels*4 , in_channels *4  ,1)
        self.contras1 = ExpansionContrastModule(in_channels=in_channels * 1, out_channels=in_channels * 1,
                                                width=img_size // 1, height=img_size // 1, shifts=[1, 3])
        self.contras2 = ExpansionContrastModule(in_channels=in_channels * 2, out_channels=in_channels * 2,
                                                width=img_size // 2, height=img_size // 2, shifts=[1, 3])
        self.contras3 = ExpansionContrastModule(in_channels=in_channels * 4, out_channels=in_channels * 4,
                                                width=img_size // 4, height=img_size // 4, shifts=[1, 3])
        self.contras4 = ExpansionContrastModule(in_channels=in_channels * 8, out_channels=in_channels * 8,
                                                width=img_size // 8, height=img_size // 8, shifts=[1, 3])
        self.AGCB = AGCB_Element(planes=in_channels * 8)

        # self.decoder6 = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels*4,out_channels=in_channels*4,kernel_size=2,stride=2),CBN(in_channels*4,in_channels*4,kernel_size=1))
        self.decoder5 = UpBlock_attention(in_channels * 16, in_channels * 8, nb_Conv=2)
        self.decoder4 = UpBlock_attention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.decoder3 = UpBlock_attention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.decoder2 = UpBlock_attention(in_channels * 4, in_channels, nb_Conv=2)
        self.decoder1 = UpBlock_attention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))

    def _make_layer(self, block, input_channels, output_channels,  ratio, patch, isdct,num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels, ratio, patch, isdct))
        for _ in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels, ratio, patch, isdct))
        return nn.Sequential(*layers)

    def forward(self, x):
        # encoder 输入x为1*256*256->16*256*256
        x1 = self.inc(x)
        # 16*256*256->32*128*128
        x2 = self.encoder1(self.pool(x1))
        # 32*128*128->64*64*64
        x3 = self.encoder2(self.pool(x2))
        # 64*64*64->128*32*32
        x4 = self.encoder3(self.pool(x3))
        # 128*32*32->128*16*16
        d5 = self.encoder4(self.pool(x4))

        # Transfor_layer
        # 1*512*512->16*512*512
        c1 = self.contras1(x1)
        # 16*512*512->32*256*256
        c2 = self.contras2(x2)
        # 32*256*256->64*128*128
        c3 = self.contras3(x3)
        # 64*128*128->128*64*64
        c4 = self.contras4(x4)
        d5 = self.AGCB(d5)
        # decoder
        # 64*64*64
        d4 = self.decoder4(d5, c4, x4)
        # 32*128*128
        d3 = self.decoder3(d4, c3, x3)
        # 16*256*256
        d2 = self.decoder2(d3, c2, x2)
        # 1*512*512
        out = self.outc(self.decoder1(d2, c1, x1))
        return out.sigmoid()
