import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=True,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        # self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.depthwise.weight, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.pointwise.weight, a=math.sqrt(5))
        nn.init.constant_(self.depthwise.bias, 0)
        nn.init.constant_(self.pointwise.bias, 0)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ConvExpert(nn.Module):
    def __init__(
        self, r, kernel_size=3, use_norm=True, activation="gelu", temperature=1.0
    ):
        super().__init__()
        self.r = r
        self.kernel_size = kernel_size
        self.use_norm = use_norm
        self.activate = activate
        self.temperature = temperature

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

        self.conv1 = DWConv(
            r, r, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
        self.conv2 = DWConv(
            r, r, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

        self.dropout = nn.Dropout(0.1)

        if use_norm:
            self.norm1 = nn.LayerNorm(r)
            self.norm2 = nn.LayerNorm(r)

        self._initialize_parameters()

    def _initialize_parameters(self):
        if self.use_norm and hasattr(self, "norm1"):
            nn.init.ones_(self.norm1.weight)
            nn.init.zeros_(self.norm1.bias)
        if self.use_norm and hasattr(self, "norm2"):
            nn.init.ones_(self.norm2.weight)
            nn.init.zeros_(self.norm2.bias)

    def forward(self, x, scale):
        B, N, r = x.shape
        H = W = int(N**0.5)

        x_2d = x.permute(0, 2, 1).reshape(B, r, H, W).contiguous()

        x_scale1 = F.interpolate(
            x_2d, scale_factor=scale, mode="bilinear", align_corners=False
        )
        x_conv1 = self.dropout(self.act(self.conv1(x_scale1)))
        x_conv1 = F.interpolate(
            x_conv1, scale_factor=(H, W), mode="bilinear", align_corners=False
        )

        x_out1 = x_conv1

        x_scale2 = F.interpolate(
            x_2d, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        x_conv2 = self.dropout(self.act(self.conv2(x_scale2)))
        x_conv2 = F.interpolate(
            x_conv2, scale_factor=(H, W), mode="bilinear", align_corners=False
        )

        x_out2 = x_conv2
        if self.use_norm:
            x_out1 = x_out1.permute(0, 2, 3, 1).contiguous()
            x_out2 = x_out2.permute(0, 2, 3, 1).contiguous()
            x_out1 = self.norm1(x_out1)
            x_out2 = self.norm2(x_out2)
            x_out1 = x_out1.permute(0, 3, 1, 2).contiguous()
            x_out2 = x_out2.permute(0, 3, 1, 2).contiguous()

        x_out = x_out1 + x_out2

        x_out = x_out.reshape(B, r, N).permute(0, 2, 1).contiguous()

        return x_out
