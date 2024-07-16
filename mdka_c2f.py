import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.block import *
from .modules.conv import *
from .modules.dka_conv import *


class MDKA_Conv(nn.Module):
    def __init__(self, dim, act=True) -> None:
        super().__init__()

        self.conv_3x3 = Conv(dim, dim // 2, 3, act=act)

        self.conv_3x3_d1 = Conv(dim // 2, dim, 3, d=1, act=act)
        self.conv_3x3_d3 = DKA_Conv_Block(dim // 2, 5)
        self.conv_3x3_d5 = DKA_Conv_Block(dim // 2, 7)

        self.conv_1x1 = Conv(dim * 2, dim, k=1, act=act)

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out



class MDKA_C2f(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MDKA_Conv(self.c) for _ in range(n))