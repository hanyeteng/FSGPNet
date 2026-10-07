import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalBlock(nn.Module):

    def __init__(self, planes, reduce_ratio=8):
        super().__init__()

        inter_planes = planes // reduce_ratio

        self.query_conv = nn.Conv2d(planes, inter_planes, 1)
        self.key_conv = nn.Conv2d(planes, inter_planes, 1)
        self.value_conv = nn.Conv2d(planes, planes, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x):

        B,C,H,W = x.shape

        q = self.query_conv(x).view(B,-1,H*W).transpose(1,2)
        k = self.key_conv(x).view(B,-1,H*W)
        v = self.value_conv(x).view(B,-1,H*W)

        attn = torch.bmm(q,k)
        attn = torch.softmax(attn,dim=-1)

        out = torch.bmm(v,attn.transpose(1,2))
        out = out.view(B,C,H,W)

        return self.gamma*out + x



class GCA_Element(nn.Module):
    def __init__(self, planes, scale, reduce_ratio_nl, att_mode='origin'):
        super(GCA_Element, self).__init__()
        assert att_mode in ['origin', 'post']

        self.att_mode = att_mode
        if att_mode == 'origin':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
            self.sigmoid = nn.Sigmoid()
        elif att_mode == 'post':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=1)
            self.conv_att = nn.Sequential(
                nn.Conv2d(planes, planes // 4, kernel_size=1),
                nn.BatchNorm2d(planes // 4),
                nn.ReLU(True),

                nn.Conv2d(planes // 4, planes, kernel_size=1),
                nn.BatchNorm2d(planes),
            )
            self.sigmoid = nn.Sigmoid()
        else:
            raise NotImplementedError

    def forward(self, x):
        batch_size, C, height, width = x.size()

        if self.att_mode == 'origin':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = F.interpolate(gca, [height, width], mode='bilinear', align_corners=True)
            gca = self.sigmoid(gca)
        elif self.att_mode == 'post':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = self.conv_att(gca)
            gca = F.interpolate(gca, [height, width], mode='bilinear', align_corners=True)
            gca = self.sigmoid(gca)
        else:
            raise NotImplementedError
        return gca



class conv_block(nn.Module):

    def __init__(self,in_features,out_features,kernel_size=3,stride=1,padding=1,
                 dilation=1,norm_type='bn',activation=True,use_bias=True,groups=1):

        super().__init__()

        layers = []

        layers.append(
            nn.Conv2d(in_features,out_features,kernel_size,stride,padding,
                      dilation=dilation,bias=use_bias,groups=groups)
        )

        if norm_type=='bn':
            layers.append(nn.BatchNorm2d(out_features))

        elif norm_type=='gn':
            layers.append(nn.GroupNorm(32 if out_features>=32 else out_features,out_features))

        if activation:
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self,x):
        return self.block(x)


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MDCR(nn.Module):

    def __init__(self, in_features, out_features, norm_type='bn', activation=True, rate=[1,2,3,4]):
        super().__init__()

        self.block1 = conv_block(in_features//4,out_features//4,padding=rate[0],dilation=rate[0],
                                 norm_type=norm_type,activation=activation,groups=in_features//4)

        self.block2 = conv_block(in_features//4,out_features//4,padding=rate[1],dilation=rate[1],
                                 norm_type=norm_type,activation=activation,groups=in_features//4)

        self.block3 = conv_block(in_features//4,out_features//4,padding=rate[2],dilation=rate[2],
                                 norm_type=norm_type,activation=activation,groups=in_features//4)

        self.block4 = conv_block(in_features//4,out_features//4,padding=rate[3],dilation=rate[3],
                                 norm_type=norm_type,activation=activation,groups=in_features//4)

        self.out_s = conv_block(
            in_features=4,
            out_features=4,
            kernel_size=1,
            padding=0,
            norm_type=norm_type,
            activation=activation
        )

    def forward(self,x):

        x1,x2,x3,x4 = torch.chunk(x,4,dim=1)

        x1 = self.block1(x1)
        x2 = self.block2(x2)
        x3 = self.block3(x3)
        x4 = self.block4(x4)

        B,C,H,W = x1.shape

        x = torch.stack([x1,x2,x3,x4],dim=2)  # B C 4 H W

        x = x.view(B*C,4,H,W)

        x = self.out_s(x)

        x = x.view(B,C,4,H,W)

        x = x.permute(0,2,1,3,4).reshape(B,4*C,H,W)  # 恢复4C

        return x


class AGCB_Element(nn.Module):

    def __init__(self,planes,scale=2,reduce_ratio_nl=16,att_mode='post'):
        super().__init__()

        self.attention = GCA_Element(planes,scale,reduce_ratio_nl,att_mode)

        self.mdcr = MDCR(planes,planes)

        self.se = SE_Block(planes)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.out_conv = nn.Conv2d(planes,planes,3,1,1)

    def forward(self,x):

        gca = self.attention(x)

        context = self.mdcr(x)

        context = context * gca

        context = self.se(context)

        context = self.gamma * context + x

        return self.out_conv(context)
