
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.block import *
from models.networks.normalization import spectral_norm

class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
        fout = opt.output_nc
        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            self.fc = nn.Conv2d(1, 16 * nf, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf


        self.conv_img = nn.Conv2d(final_nc, fout, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

        self.prev_z = 0.

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input
        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)
        x = self.up(x)
        x = self.G_middle_0(x, seg)


        x = self.G_middle_1(x, seg)
        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x


class HPUGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
        fout = opt.output_nc
        self.sw, self.sh = self.compute_latent_vector_size(opt)

        self.en_0 = SPADEResnetBlock(1 * nf, 2 * nf, opt)
        self.en_1 = SPADEResnetBlock(2 * nf, 4 * nf, opt)
        self.en_2 = SPADEResnetBlock(4 * nf, 8 * nf, opt)
        self.en_3 = SPADEResnetBlock(8 * nf, 16 * nf, opt)

        if opt.use_vae:
            self.fc = nn.Conv2d(1, 16 * nf, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.de_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.de_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.de_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.de_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        self.conv_img = nn.Conv2d(final_nc, fout, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d

        self.prev_z = 0.


        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):

        x = self.en_0(input)
        skip0 = x
        x = self.down(x)
        x = self.en_1(x)
        skip1 = x
        x = self.down(x)
        x = self.en_2(x)
        skip2 = x
        x = self.down(x)
        x = self.en_3(x)
        skip3 = x
        x = self.down(x)
        x = self.en_1(x)

        seg = input
        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)
        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up_0(x, seg)
        x = self.up_1(x, seg)
        x = self.up_2(x, seg)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            '--resnet_n_downsample',
            type=int,
            default=4,
            help='number of downsampling layers in netG')
        parser.add_argument(
            '--resnet_n_blocks',
            type=int,
            default=9,
            help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + \
            (1 if opt.contain_dontcare_label else 0) + \
            (0 if opt.no_instance else 1)
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)


class SCFEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')

        return parser

    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        nf = opt.ngf
        fout = opt.output_nc
        fin = 1

        self.gate1 = GATEconv(fin, nf, 7, 2, use_lrn=False)
        self.gate2 = GATEconv(nf, nf * 2, 5, 2)
        self.gate3 = GATEconv(nf * 2, nf * 4, 5, 2)
        self.gate4 = GATEconv(nf * 4, nf * 8, 3, 2)
        self.gate5 = GATEconv(nf * 8, nf * 8, 3, 2)
        self.gate6 = GATEconv(nf * 8, nf * 8, 3, 2)
        self.gate7 = GATEconv(nf * 8, nf * 8, 3, 2)

        self.dlt1 = GATEconv(nf * 8, nf * 8, 3, 1, rate=2)
        self.dlt2 = GATEconv(nf * 8, nf * 8, 3, 1, rate=4)
        self.dlt3 = GATEconv(nf * 8, nf * 8, 3, 1, rate=8)
        self.dlt4 = GATEconv(nf * 8, nf * 8, 3, 1, rate=16)

        self.gate8 = GATEdeconv(nf * 8, nf * 8)
        self.conv8 = GATEconv(nf * 16, nf * 8, 3, 1)
        self.gate9 = GATEdeconv(nf * 8, nf * 8)
        self.conv9 = GATEconv(nf * 16, nf * 8, 3, 1)
        self.gate10 = GATEdeconv(nf * 8, nf * 8)
        self.conv10 = GATEconv(nf * 16, nf * 8, 3, 1)
        self.gate11 = GATEdeconv(nf * 8, nf * 4)
        self.conv11 = GATEconv(nf * 8, nf * 4, 3, 1)
        self.gate12 = GATEdeconv(nf * 4, nf * 2)
        self.conv12 = GATEconv(nf * 4, nf * 2, 3, 1)
        self.gate13 = GATEdeconv(nf * 2, nf * 1)
        self.conv13 = GATEconv(nf * 2, nf * 1, 3, 1)
        self.gate14 = GATEdeconv(nf * 1, fout)
        self.conv14 = GATEconv(fout * 2, fout, 3, 1, use_lrn=False, activation=False)

    def forward(self, input, z=None):

        x1, mask1 = self.gate1(input)
        x2, mask2 = self.gate2(x1)
        x3, mask3 = self.gate3(x2)
        x4, mask4 = self.gate4(x3)
        x5, mask5 = self.gate5(x4)
        x6, mask6 = self.gate6(x5)
        x7, mask7 = self.gate7(x6)

        x7, _ = self.dlt1(x7)
        x7, _ = self.dlt2(x7)
        x7, _ = self.dlt3(x7)
        x7, _ = self.dlt4(x7)

        x8, _ = self.gate8(x7)
        x8 = torch.cat([x6, x8], dim=1)
        x8, mask8 = self.conv8(x8)
        x9, _ = self.gate9(x8)
        x9 = torch.cat([x5, x9], dim=1)
        x9, mask9 = self.conv9(x9)
        x10, _ = self.gate10(x9)
        x10 = torch.cat([x4, x10], dim=1)
        x10, mask10 = self.conv10(x10)
        x11, _ = self.gate11(x10)
        x11 = torch.cat([x3, x11], dim=1)
        x11, mask11 = self.conv11(x11)
        x12, _ = self.gate12(x11)
        x12 = torch.cat([x2, x12], dim=1)
        x12, mask12 = self.conv12(x12)
        x13, _ = self.gate13(x12)
        x13 = torch.cat([x1, x13], dim=1)
        x13, mask13 = self.conv13(x13)
        x14, _ = self.gate14(x13)
        x14 = torch.cat([input, x14], dim=1)
        x14, mask14 = self.conv14(x14)
        output = torch.tanh(x14)

        return output#, mask14

    
from functools import partial

class BridgeUnet(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        input_nf =1
        output_nf =1
        nf = 32
        down = nn.MaxPool2d()
        cat = partial(torch.cat,dim=1)
        
        self.stage1 = nn.ModuleList([
            nn.Sequential([Block(input_nf,nf), Block(nf,nf)]),
            nn.Sequential([Block(nf,nf*2), Block(nf*2,nf*2)]),
            nn.Sequential([Block(nf*2,nf*4), Block(nf*4,nf*4)]),
            nn.Sequential([Block(nf*4,nf*8,activation=nn.ReLU(True)),
                           Block(nf*8,nf*8,activation=nn.ReLU(True))]),
            nn.Sequential([Block(nf*8,nf*16,activation=nn.ReLU(True)),
                           Block(nf*16,nf*16,activation=nn.ReLU(True))])
                                    ])
                                     
        self.stage2 = nn.ModuleList([
            nn.Sequential([Block(nf*16,nf*16,activation=nn.ReLU(True)),
                           Block(nf*16,nf*16,activation=nn.ReLU(True))]),
            nn.Sequential([Block(nf*16,nf*8), Block(nf*8, nf*8)]),
            nn.Sequential([Block(nf*8,nf*4), Block(nf*4,nf*4)]),
            nn.Sequential([Block(nf*4,nf*2), Block(nf*2,nf*2)]),
            nn.Sequential([Block(nf*2,nf*1), Block(nf*1,nf*1)])
                                    ])
        
        #U-net 1 upsample
        self.up1 = nn.ModuleList([nn.ConvTranspose2d(nf*16,nf*8,2,stride=2),
                                  nn.ConvTranspose2d(nf*8,nf*4,2,stride=2),
                                  nn.ConvTranspose2d(nf*4,nf*2,2,stride=2),
                                  nn.ConvTranspose2d(nf*2,nf*1,2,stride=2),
                                  nn.Linear(1,1)
                                 ])
        
        self.stage3 = nn.ModuleList([
            nn.Sequential([Block(nf,nf), Block(nf,nf)]),
            nn.Sequential([Block(nf*2,nf*2), Block(nf*2,nf*2)]),
            nn.Sequential([Block(nf*4,nf*4), Block(nf*4,nf*4)]),
            nn.Sequential([Block(nf*8,nf*8,activation=nn.ReLU(True)),
                           Block(nf*8,nf*8,activation=nn.ReLU(True))]),
            nn.Sequential([Block(nf*16,nf*16,activation=nn.ReLU(True)),
                           Block(nf*16,nf*16,activation=nn.ReLU(True))])
        ])
        
        
        self.stage4 = nn.ModuleList([
            nn.Sequential([Block(nf*8,nf*16,activation=nn.ReLU(True)),
                           Block(nf*16,nf*16,activation=nn.ReLU(True))]),
            nn.Sequential([Block(nf*16,nf*8), Block(nf*8, nf*8)]),
            nn.Sequential([Block(nf*8,nf*4), Block(nf*4,nf*4)]),
            nn.Sequential([Block(nf*4,nf*2), Block(nf*2,nf*2)]),
            nn.Sequential([Block(nf*2,nf*1), Block(nf*1,nf*1)])
                          ])
            
        # U-net
        self.up2 = nn.ModuleList([nn.ConvTranspose2d(nf*16,nf*8,2,stride=2),
                                  nn.ConvTranspose2d(nf*8,nf*4,2,stride=2),
                                  nn.ConvTranspose2d(nf*4,nf*2,2,stride=2),
                                  nn.ConvTranspose2d(nf*2,nf*1,2,stride=2),
                                  nn.Linear(1,1)
                                 ])
        self.end = nn.Conv2d(nf,1,1)
    def forward(self, x):
        skip1=[]
        skip2=[]
        skip3=[]
        for layer in self.stage1:
            x = layer(x)
            skip1.append(x)
            x = self.down(x)
        skip1.append(False)
        skip1.reverse()
        for layer, up, skip in zip(self.stage2, self.up1, skip1):
            if skip:
                x = cat([skip, x])
            x = layer(x)
            skip2.append(x)
            if is_instance(up, nn.ConvTranspose2d):
                x = up(x)
        skip2.append(False)
        skip2.reverse()
        skip1.reverse()
        for layer, addskip, skip in (self.stage3, skip1, skip2):
            if skip:
                x = cat([skip, x])
            x = layer(x)
            if addskip:
                skip3.append(x+addskip)
            x = self.down(x)
        skip3.append(False)
        skip3.reverse()
        for layer, up in zip(self.stage4, self.up2, skip3):
            if skip:
                x = cat([skip, x])
            x = layer(x)
            if is_instance(up, nn.ConvTranspose2d):
                x = up(x)
        x = self.end(x)
        return x
            
            