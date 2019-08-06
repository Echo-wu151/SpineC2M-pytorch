
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE
from models.networks.normalization import get_nonspade_norm_layer
from torch.nn import Parameter as P


class ATTBlock(nn.Module):

    def __init__(self, opt):
        super().__init__()
        
        self.ksize = 7
        pw = int( (self.ksize-1)//2)
        self.bsize = opt.batchSize // len(opt.gpu_ids)

        self.pos = torch.linspace(-4,
                                  4,
                                  steps=(self.ksize)**2)[:,
                                                       None,
                                                       None]
        self.unfold = nn.Unfold(self.ksize, dilation=1, padding=1, stride=1)
            
        self.fc = nn.Linear(49,49) 

    def forward(self, x, z=None):
        size = x.size()

        v = self.unfold(x).view(size(0), size(1), -1, size(2)*size(3)) 
        x = x.view(size(0), size(1), -1, size(2)*size(3)) 
        p = p.repeat(size(0), size(1), 1, size(2)*size(3))

        xv = torch.matmul(v,x)
        xp = torch.matmul(x,p)
        h = xv + xp
        h = h.view(-1, h.size(2))
        h = self.fc(h)
        h = h.view(*size)

        return x
class Contextual_Attention_Module(nn.Module):
    def __init__(self, in_ch, out_ch, rate=2, stride=1):
        super(Contextual_Attention_Module, self).__init__()
        self.rate = rate
        self.padding = nn.ReflectionPad2d(1)
        self.up_sample = nn.UpsamplingNearest2d(scale_factor=self.rate)
        layers = []
        for i in range(2):
            layers.append(Conv(in_ch, out_ch))
        self.out = nn.Sequential(*layers)
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, f, b, mask=None, ksize=3, softmax_scale=10., training=True):

        # get shapes
        raw_fs = f.size() # B x 128 x 64 x 64
        raw_int_fs = list(f.size())
        raw_int_bs = list(b.size())

        # extract patches from background with stride and rate
        kernel = 2*self.rate
        raw_w = self.extract_patches(b, kernel=kernel, stride=self.rate)
        raw_w = raw_w.contiguous().view(raw_int_bs[0], -1, raw_int_bs[1], kernel, kernel) # B*HW*C*K*K (B, 32*32, 128, 4, 4)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = down_sample(f, scale_factor=1/self.rate, mode='nearest')
        b = down_sample(b, scale_factor=1/self.rate, mode='nearest')
        fs = f.size() # B x 128 x 32 x 32
        int_fs = list(f.size())
        f_groups = torch.split(f, 1, dim=0) # Split tensors by batch dimension; tuple is returned

        # from b(B*H*W*C) to w(b*k*k*c*h*w)
        bs = b.size() # B x 128 x 32 x 32
        int_bs = list(b.size())
        w = self.extract_patches(b)
        w = w.contiguous().view(int_fs[0], -1,int_fs[1]*ksize*ksize) # B*HW*C*K*K (B, 32*32, 128, 3, 3)

        # process mask
        if mask is not None:
            mask = down_sample(mask, scale_factor=1./self.rate, mode='nearest')
        else:
            mask = torch.zeros([1, 1, bs[2], bs[3]])

        m = self.extract_patches(mask)
        m = m.contiguous().view(int_fs[0], 1, -1, ksize, ksize)  # B*C*HW*K*K
        m = m[0] # (1, 32*32, 3, 3)
        m = reduce_mean(m) # smoothing, maybe
        mm = m.eq(0.).float() # (1, 32*32, 1, 1)       
        
        w_groups = torch.split(w, 1, dim=0) # Split tensors by batch dimension; tuple is returned
        raw_w_groups = torch.split(raw_w, 1, dim=0) # Split tensors by batch dimension; tuple is returned
        y = []
        scale = softmax_scale
        
        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):

            # conv for compare
            wi = wi[0]
            key = fs[2]
            xi = F.unfold(xi,(3,3))     #  1 128*9 32*32
            wi = wi.view(wi.size(0),-1).unsqueeze(-1)  # 32*32 12*89 1

            yi = (xi - wi).matrix_power(2).sum(1).sqrt()  # 32*32 32*32
            yi = yi.view(1,key**2, key,key) # yi => (B=1, C=32*32, H=32, W=32)
            yi = torch.tanh(-(yi-yi.mean()/yi.std()))

            # softmax to match
            yi = yi * mm  # mm => (1, 32*32, 1, 1)
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  # mask

            # deconv for patch pasting
            # 3.1 paste center
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4. # (B=1, C=128, H=64, W=64)
            y.append(yi)

        y = torch.cat(y, dim=0) # back to the mini-batch
        y.contiguous().view(raw_int_fs)

        return self.out(y)

    # padding1(16 x 128 x 64 x 64) => (16 x 128 x 64 x 64 x 3 x 3)
    def extract_patches(self, x, kernel=3, stride=1):
        x = self.padding(x)
        all_patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        return all_patches
    
class SPADEBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        spade_config_str = opt.norm_G

        fmid = min(fin, fout)

        # create conv layers
        self.block_0 = Block(fin, fmid, kernel_size=3, padding=1)
        self.block_1 = Block(fmid, fout, kernel_size=3, padding=1)

        if self.learned_shortcut:
            self.conv_s = conv(fin, fout, kernel_size=1, bias=False)

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        x = self.conv_0(x, seg)
        x = self.conv_1(x, seg)
        out = x_s + x

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(x, seg)
        else:
            x_s = x
        return x_s


class Block(nn.Module):
    def __init__(self, fin, fout, norm=nn.BatchNorm2d,
                 activation=nn.ELU(inplace=True), 
                 conv=spectral_norm(nn.Conv2d),kernel_size=3, 
                 stride=1, dilation=1, groups=1, bias=True,
                 padding_mode=nn.ReflectionPad2d):
        super().__init__()
        
        if not activation: activation = nn.Identity()
        if not norm: norm = nn.Identity
        if not layer: layer = nn.Identity
        if not padding_mode = nn.ZeroPad2d
        pw = (kernel_size - 1) // 2
        
        self.conv_block = nn.Sequential(
            padding_mode(pw)
            conv(fin, fout, kernel_size=kernel_size, padding=pw,
                      stride=stride, dilation=dilation, 
                      groups=groups, bias=bias),
            norm(fout),
            activation
        )

    def forward(self, x, *args):
        return self.conv_block(x)
    
class GateBlock(Block):
    def __init__(self):
        super().__init__()
        self.gate = conv(fin, fout, kernel_size=kernel_size, padding=pw,
                      stride=stride, dilation=dilation, 
                      groups=groups, bias=bias)

    def forward(self, x):
        g = torch.sigmoid(self.gate(x))
        x = self.conv_block(x)
        x = x * g
        return x
    
class BottleNeckBlock(nn.Module):
    def __init__(self, channel_factor = 4):
        super().__init__()
        
        fmid = fout//channel_factor
        
        self.neck1 = Block(fin, fmid, kernel_size=kernel_size, padding=pw,
                      stride=stride, dilation=dilation, 
                      groups=groups, bias=bias)
        
        self.neck2 = Block(fmid, fout, kernel_size=kernel_size, padding=pw,
                      stride=stride, dilation=dilation, 
                      groups=groups, bias=bias)

    def forward(self, x):
        x = self.neck1(x)
        x = self.conv_block(x)
        x = self.neck2(x)
        return x
    
    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s
    
class ResBlock(nn.Module):
    def __init__(self,cut = Block):
        super().__init__()
        
        fmid = fout//channel_factor
        
        self.cut = Block(fin, fmid, kernel_size=kernel_size, padding=pw,
                      stride=stride, dilation=dilation, 
                      groups=groups, bias=bias)
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.shortcut(x)
        
        return x
    
    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.cut(x)
        else:
            x_s = x
        return x_s

class NonLocalBlock(nn.Module):
    def __init__(self, ch, name='attention', conv=spectral_norm(nn.Conv2d)):
        super(NonLocal, self).__init__()

        self.ch = ch

        self.theta = conv(ch, ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = conv(ch, ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = conv(ch, ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = conv(ch // 2, ch, kernel_size=1, padding=0, bias=False)

        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])

        # Perform reshapes
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)

        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1,
                                                           self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x