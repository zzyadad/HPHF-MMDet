import math
import torch.nn as nn
import torch
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    
    
class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
        
class DSConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True) -> None:
        super().__init__()
        
        self.dwconv = DWConv(c1, c1, 3)
        self.pwconv = Conv(c1, c2, 1)
    
    def forward(self, x):
        return self.pwconv(self.dwconv(x))
    
class SeparateQKV(nn.Module):
    def __init__(self, in_C, out_C):
        super(SeparateQKV, self).__init__()
        
        self.RGB_Q = DSConv(out_C, out_C, 3)
        self.INF_Q = DSConv(out_C, out_C, 3)
        
        self.RGB_K = DSConv(out_C, out_C, 3)
        self.RGB_V = DSConv(out_C, out_C, 3)
        self.INF_K = DSConv(out_C, out_C, 3)
        self.INF_V = DSConv(out_C, out_C, 3)
        
        self.Second_reduce = DSConv(in_C, out_C, 3)
        
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.temperature = nn.Parameter(torch.ones(1))
        self.register_buffer("attn_scale", torch.tensor(out_C ** -0.5, dtype=torch.float32))
        self.eps = 1e-6

        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, y):
        m_batchsize, C, height, width = x.size()
        N = height * width

        RGB_Q = self.RGB_Q(x)
        INF_Q = self.INF_Q(y)

        INF_K = self.INF_K(y)
        INF_V = self.INF_V(y)

        RGB_Q_flat = F.normalize(RGB_Q.view(m_batchsize, -1, N), dim=1)
        INF_K_flat = F.normalize(INF_K.view(m_batchsize, -1, N), dim=1)

        attn_temp = self.temperature.abs() + self.eps
        RGB_attn = torch.bmm(INF_K_flat.permute(0, 2, 1), RGB_Q_flat)
        RGB_attn = self.softmax(RGB_attn * self.attn_scale / attn_temp)

        RGB_refine = torch.bmm(INF_V.view(m_batchsize, -1, N), RGB_attn.permute(0, 2, 1))
        RGB_refine = RGB_refine.view(m_batchsize, -1, height, width)
        rgb_gate = torch.sigmoid(self.gamma1)
        RGB_refine = rgb_gate * RGB_refine + (1.0 - rgb_gate) * y

        RGB_K = self.RGB_K(x)
        RGB_V = self.RGB_V(x)

        RGB_K_flat = F.normalize(RGB_K.view(m_batchsize, -1, N), dim=1)
        INF_Q_flat = F.normalize(INF_Q.view(m_batchsize, -1, N), dim=1)

        INF_attn = torch.bmm(RGB_K_flat.permute(0, 2, 1), INF_Q_flat)
        INF_attn = self.softmax(INF_attn * self.attn_scale / attn_temp)

        INF_refine = torch.bmm(RGB_V.view(m_batchsize, -1, N), INF_attn.permute(0, 2, 1))
        INF_refine = INF_refine.view(m_batchsize, -1, height, width)
        inf_gate = torch.sigmoid(self.gamma2)
        INF_refine = inf_gate * INF_refine + (1.0 - inf_gate) * x

        out = self.Second_reduce(torch.cat([RGB_refine, INF_refine], dim=1))

        return out


class CM_CHF(nn.Module):
    def __init__(self, Channel):
        super(CM_CHF, self).__init__()
        self.obj_fuse = SeparateQKV(Channel * 2, Channel)
        
    def forward(self, data):
        rgb, depth = data
        out = self.obj_fuse(rgb, depth)
        return out