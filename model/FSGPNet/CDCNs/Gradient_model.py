import torch
from torch import nn
import torch.nn.functional as F
import math


class PhaseEnhancedGaborFilter(nn.Module):

    def __init__(self, in_channels, kernel_size, orientations=8,
                 n_freqs=1, init_phase_weight=2.7):

        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.orientations = orientations
        self.n_freqs = n_freqs

        # 相位权重
        self.log_phase_weight = nn.Parameter(
            torch.tensor(math.log(init_phase_weight), dtype=torch.float32)
        )

        # Gabor参数
        self.log_lambd = nn.Parameter(torch.zeros(orientations, n_freqs))
        self.log_sigma = nn.Parameter(torch.zeros(orientations, n_freqs))
        self.gamma = nn.Parameter(torch.ones(orientations, n_freqs) * 0.5)

        self._initialize_parameters()

        # ===== 预生成坐标网格 =====
        center = kernel_size // 2
        coords = torch.arange(-center, center + 1).float()

        y_grid, x_grid = torch.meshgrid(coords, coords, indexing='ij')

        # meshgrid returns expanded views; checkpoint loading needs owned storage.
        self.register_buffer("x_grid", x_grid.clone())
        self.register_buffer("y_grid", y_grid.clone())

    def _initialize_parameters(self):

        init_lambd = self.kernel_size * 0.5
        init_sigma = 0.56 * init_lambd

        for i in range(self.orientations):
            for j in range(self.n_freqs):

                scale = 1.0 / (2 ** j)

                self.log_lambd.data[i, j] = math.log(init_lambd * scale)
                self.log_sigma.data[i, j] = math.log(init_sigma * scale)

    @property
    def phase_weight(self):
        return F.softplus(self.log_phase_weight)

    def build_gabor_kernels(self, device):

        x = self.x_grid.to(device)
        y = self.y_grid.to(device)

        O = self.orientations
        Freq = self.n_freqs
        K = self.kernel_size

        # ===== orientation tensor =====
        theta = torch.arange(O, device=device) * math.pi / O
        cos_theta = torch.cos(theta).view(O, 1, 1)
        sin_theta = torch.sin(theta).view(O, 1, 1)

        # ===== 坐标旋转 =====
        x_theta = cos_theta * x + sin_theta * y
        y_theta = -sin_theta * x + cos_theta * y

        x_theta = x_theta.unsqueeze(1)
        y_theta = y_theta.unsqueeze(1)

        # ===== 参数 =====
        lambd = torch.exp(self.log_lambd).view(O, Freq, 1, 1)
        sigma = torch.exp(self.log_sigma).view(O, Freq, 1, 1)
        gamma = torch.clamp(self.gamma, 0.1, 1.0).view(O, Freq, 1, 1)

        # ===== Gabor =====
        exponent = -(x_theta**2 + (gamma**2)*(y_theta**2)) / (2*sigma**2)
        gauss = torch.exp(exponent)

        real = gauss * torch.cos(2*math.pi*x_theta/lambd)
        imag = gauss * torch.sin(2*math.pi*x_theta/lambd)

        # magnitude = torch.abs(real)
        magnitude = torch.sqrt(real**2 + imag**2)

        # ===== 相位增强 =====
        phase_grad = 2*math.pi/lambd
        enhanced = magnitude * (1 + self.phase_weight * phase_grad)

        # ===== 带通归一化 =====
        enhanced = enhanced - enhanced.mean(dim=(-1,-2), keepdim=True)

        norm = torch.sum(torch.abs(enhanced), dim=(-1,-2), keepdim=True)
        enhanced = enhanced / (norm + 1e-6)

        kernels = enhanced.view(O*Freq, K, K)

        return kernels

    def forward(self, x):

        B, C, H, W = x.shape
        device = x.device

        kernels = self.build_gabor_kernels(device)

        kernels = kernels.view(-1,1,self.kernel_size,self.kernel_size)
        kernels = kernels.repeat(C,1,1,1)

        padding = self.kernel_size//2

        out = F.conv2d(
            x,
            kernels,
            padding=padding,
            groups=C
        )

        return out

class ExpansionContrastModule(nn.Module):
    """
    改进版 ECM：多尺度 Gabor + 注意力
    """

    def __init__(self, in_channels, out_channels, width, height, shifts=None, kernel_sizes=[5, 7]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.num_heads = len(kernel_sizes)  # 不同 kernel size = 不同 head

        # 空间归一化
        self.width = width
        self.height = height
        self.area = width * height
        self.psi = nn.InstanceNorm2d(self.num_heads)
        self.softmax_layer = nn.Softmax(dim=-1)

        # 多尺度 CDC-Gabor 滤波器
        self.gabor_filters = nn.ModuleList()
        for size in kernel_sizes:
            gabor_filter = PhaseEnhancedGaborFilter(
                in_channels=in_channels,
                kernel_size=size,
                orientations=8
            )
            self.gabor_filters.append(gabor_filter)

        # Q,K,V 映射
        self.query_convs = nn.ModuleList()
        self.key_convs = nn.ModuleList()
        self.value_convs = nn.ModuleList()

        self.hidden_channels = max(1, in_channels // self.num_heads)
        for i in range(self.num_heads):
            self.query_convs.append(
                nn.Conv2d(in_channels, self.hidden_channels, kernel_size=1, stride=1, bias=False)
            )
            self.key_convs.append(
                nn.Conv2d(in_channels * 8, self.hidden_channels * 8, kernel_size=1, stride=1, bias=False)
            )
            self.value_convs.append(
                nn.Conv2d(in_channels * 8, self.hidden_channels * 8, kernel_size=1, stride=1, bias=False)
            )

        # 输出卷积
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def extract_layer(self, cen):
        """提取多尺度 Gabor 特征"""
        b, c, h, w = cen.shape
        surrounds_keys, surrounds_querys, surrounds_values = [], [], []

        for i in range(self.num_heads):
            gabor_output = self.gabor_filters[i](cen)
            surrounds_querys.append(self.query_convs[i](cen))
            surrounds_keys.append(self.key_convs[i](gabor_output))
            surrounds_values.append(self.value_convs[i](gabor_output))

        surrounds_keys = torch.stack(surrounds_keys, dim=1).view(b, self.num_heads, -1, h * w)
        surrounds_querys = torch.stack(surrounds_querys, dim=1).view(b, self.num_heads, -1, h * w)
        surrounds_values = torch.stack(surrounds_values, dim=1).view(b, self.num_heads, -1, h * w)
        return surrounds_keys, surrounds_querys, surrounds_values

    def forward(self, cen):
        b, _, h, w = cen.shape
        deltas_keys, deltas_querys, deltas_values = self.extract_layer(cen)

        # 归一化
        deltas_keys = F.normalize(deltas_keys, dim=-1).transpose(-2, -1)
        deltas_querys = F.normalize(deltas_querys, dim=-1)

        # 注意力
        d_k = deltas_querys.size(-1)
        attention_scores = torch.matmul(deltas_querys, deltas_keys) / math.sqrt(d_k)
        attention_weights = self.softmax_layer(self.psi(attention_scores))

        out = torch.matmul(attention_weights, deltas_values)
        out = out.view(b, self.in_channels, h, w)
        return self.out_conv(out)
