import torch
import models.networks as networks
import util.util as util
import torch.nn.functional as F


class Model(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG1, self.netD1, self.netG2, self.netD2 = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if not opt.no_L1_loss:
                self.criterionL1 = torch.nn.L1Loss()
            if not opt.no_Histogram_loss:
                self.criterionMI = networks.HistogramLoss()

        self.count = 0.
        self.eps = 0.
    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):

        input_semantics, real_image = data['CT'], data['MR']

        if mode == 'generator_f':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, is_forward=True)
            return g_loss, generated

        elif mode == 'discriminator_f':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, is_forward=True)
            return d_loss
        
        elif mode == 'generator_b':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, is_forward=False)
            return g_loss, generated

        elif mode == 'discriminator_b':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, is_forward=False)
            return d_loss

        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate1_fake(input_semantics, real_image)
            return fake_image

        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):

        G1_params = list(self.netG1.parameters())
        D1_params = list(self.netD1.parameters()) if opt.isTrain else None
        
        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G1 = torch.optim.Adam(G1_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D1 = torch.optim.Adam(D1_params, lr=D_lr, betas=(beta1, beta2)) if opt.isTrain else None
        
        G2_params = list(self.netG2.parameters())
        D2_params = list(self.netD2.parameters()) if opt.isTrain else None
        
        optimizer_G2 = torch.optim.Adam(G2_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D2 = torch.optim.Adam(D2_params, lr=D_lr, betas=(beta1, beta2)) if opt.isTrain else None

        return optimizer_G1, optimizer_D1, optimizer_G2, optimizer_D2

    
    def save(self, epoch):
        util.save_network(self.netG1 'G1', epoch, self.opt)
        util.save_network(self.netD1, 'D1', epoch, self.opt)
        util.save_network(self.netG2, 'G2', epoch, self.opt)
        util.save_network(self.netD2, 'D2', epoch, self.opt)

    ##########################################################################
    # Private helper methods
    ##########################################################################

    def initialize_networks(self, opt):
        netG1 = networks.define_G(opt)
        netD1 = networks.define_D(opt) if opt.isTrain else None
        netG2 = networks.define_G(opt)
        netD2 = networks.define_D(opt) if opt.isTrain else None
        if not opt.isTrain or opt.continue_train:
            netG1 = util.load_network(netG1, 'G1', opt.which_epoch, opt)
            netG2 = util.load_network(netG2, 'G2', opt.which_epoch, opt)
            if opt.isTrain:
                netD1 = util.load_network(netD1, 'D1', opt.which_epoch, opt)
                netD2 = util.load_network(netD2, 'D2', opt.which_epoch, opt)

        return netG1, netD1, netG2, netD2


    def compute_generator_loss(self, input_semantics, real_image,is_forward=True):
        # for_D : criterion for discriminator
        G_losses = {}
        if is_forward:
            fake_image = self.generate1_fake(
                input_semantics, real_image)
            pred_fake, pred_real = self.discriminate1(
            input_semantics, fake_image, real_image)
        else:
            fake_image = self.generate2_fake(
                input_semantics, real_image)
            pred_fake, pred_real = self.discriminate2(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_D=False) * self.opt.lambda_gan

        if not self.opt.no_ganFeat_loss:

            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
                # last output is the final prediction, so we exclude it
                num_outputs = len(pred_fake[i]) - 1
            for j in range(num_outputs):  # for each layer output
                unweighted_loss = self.criterionFeat(
                    pred_fake[i][j], pred_real[i][j].detach())
                GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if self.opt.plus_MI_loss:
            G_losses['MI'] = self.criterionMI(fake_image, real_image) \
                * self.opt.lambda_MI

        if self.opt.plus_L1_loss:
            G_losses['L1'] = self.criterionL1(fake_image, real_image) \
                * self.opt.lambda_L1

        size = list(fake_image.size())
        if (int(size[1]) != 3) and (not self.opt.no_vgg_loss):
            size[1] = 3
            fake_image = fake_image.expand(*size)
            real_image = real_image.expand(*size)

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        return G_losses, fake_image


    def compute_discriminator_loss(self, input_semantics, real_image,is_forward):
        D_losses = {}
        if is_forward:
            with torch.no_grad():
                fake_image, _ = self.generate1_fake(input_semantics, real_image)
                fake_image = fake_image.detach()
                fake_image.requires_grad_()
            pred_fake, pred_real = self.discriminate1(
                input_semantics, fake_image, real_image)
        else:
            with torch.no_grad():
                fake_image, _ = self.generate2_fake(input_semantics, real_image)
                fake_image = fake_image.detach()
                fake_image.requires_grad_()
            pred_fake, pred_real = self.discriminate2(
                input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_D=True) * self.opt.lambda_gan
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_D=True) * self.opt.lambda_gan

        return D_losses

    
    def generate1_fake(self, input_semantics, real_image):
        fake_image = self.netG1(input_semantics)
        return fake_image

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate1(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out = self.netD1(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    def generate2_fake(self, input_semantics, real_image):
        fake_image = self.netG2(input_semantics)
        return fake_image

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate2(self, input_semantics, fake_image, real_image):
        fake_and_real = torch.cat([fake_image, real_image], dim=0)
        discriminator_out = self.netD2(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real
    
    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if isinstance(pred, list):
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0