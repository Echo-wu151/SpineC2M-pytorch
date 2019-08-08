import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
import util.util as util


class HybridDiscriminator(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf

        self.head_aligned = nn.Sequential([Block(2, nf, kernel_size=kw, stride=2, padding=padw,norm=False)])
        self.head_unaligned = nn.Sequential([Block(1, nf, kernel_size=kw, stride=2, padding=padw,norm=False)])
        
        self.body = nn.ModuleList([Block(nf * 2 ** i, nf * 2 ** (i+1), kernel_size=kw, stride=2, padding=padw)
                    for i in range(4)])
        
        self.tail_aligned = nn.Sequential([Block(nf * 8, 1, kernel_size=kw, stride=1, padding=padw,norm=False,activation=False)])
        self.tail_unaligned = nn.Sequential([Block(nf * 8, 1, kernel_size=kw, stride=1, padding=padw,norm=False,activation=False)])


    def forward(self, input, aligned=True):
                                     
        if aligned: x = self.head_aligned(input)
        else: x = self.head_unaligned(input)
        
        for layer in self.body:
            x = layer(x)
                                                 
        if aligned: x = self.tail_aligned(x)
        else: x = self.tail_unaligned(x)

        return x
                                   
class Block(nn.Module):
    def __init__(self, fin, fout, norm=nn.InstanceNorm2d,
                 activation=nn.LeakyReLU(0.2,True), 
                 conv=spectral_norm(nn.Conv2d),kernel_size=4, 
                 stride=2, dilation=1, groups=1, bias=True,
                 padding_mode=False):
        super().__init__()
        pw = (kernel_size - 1) // 2
        if not activation: activation = nn.Identity()
        if not norm: norm = nn.Identity
        if not layer: layer = nn.Identity
        if not padding_mode: padding_mode = nn.ZeroPad2d
        
        
        self.conv_block = nn.Sequential(
            padding_mode(pw)
            conv(fin, fout, kernel_size=kernel_size, padding=0,
                      stride=stride, dilation=dilation, 
                      groups=groups, bias=bias),
            norm(fout),
            activation
        )

    def forward(self, x, *args):
        return self.conv_block(x)