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

        self.netG_for_CT, self.netD_aligned, self.netG_for_MR, self.netD_unaligned = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if not opt.no_L1_loss:
                self.criterionL1 = torch.nn.L1Loss()
            if not opt.no_SSIMLoss:
                self.criterionSSIM = networks.SSIMLoss()
            if not opt.no_GradientDifferenceLoss:
                self.criterionGDL = networks.GradientDifferenceLoss()
            if not opt.no_Histogram_loss:
                self.criterionMI = networks.HistogramLoss()
            if not opt.no_cycle_loss:
                self.criterionCYC = torch.nn.L1Loss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, CT, MR, mode):

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(CT, MR)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(CT, MR)
            return d_loss
        elif mode == 'inferenceCT':
            with torch.no_grad():
                synCT = self.generate_synCT(MR)
            return synCT
        elif mode == 'inferenceMR':
            with torch.no_grad():
                synMR = self.generate_synMR(CT)
            return synMR
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2
            
        G_for_CT_params = list(self.netG_for_CT.parameters())
        D_aligned_params = list(self.netD_aligned.parameters()) if opt.isTrain else None
        G_for_MR_params = list(self.netG_for_MR.parameters())
        D_unaligned_params = list(self.netD_unaligned.parameters()) if opt.isTrain else None
        
        optimizerG = torch.optim.Adam(G_for_CT_params+G_for_MR_params, lr=G_lr, betas=(beta1, beta2))
        optimizerD = torch.optim.Adam(D_aligned_params+D_unaligned_params, lr=D_lr, betas=(beta1, beta2)) if opt.isTrain else None
    
        return optimizerG, optimizerD

    
    def save(self, epoch):
        util.save_network(self.netG_for_CT 'G_for_CT', epoch, self.opt)
        util.save_network(self.netD_aligned, 'D_aligned', epoch, self.opt)
        util.save_network(self.netG_for_MR, 'G_for_MR', epoch, self.opt)
        util.save_network(self.netD_unaligned, 'D_unaligned', epoch, self.opt)

    ##########################################################################
    # Private helper methods
    ##########################################################################

    def initialize_networks(self, opt):
        netG_for_CT = networks.define_G(opt)
        netD_aligned = networks.define_D(opt) if opt.isTrain else None
        netG_for_MR = networks.define_G(opt)
        netD_unaligned = networks.define_D(opt) if opt.isTrain else None
        
        if not opt.isTrain or opt.continue_train:
            netG_for_CT = util.load_network(netG_for_CT, 'G_for_CT', opt.which_epoch, opt)
            netG_for_MR = util.load_network(netG_for_MR, 'G_for_MR', opt.which_epoch, opt)
            if opt.isTrain:
                netD_aligned = util.load_network(netD_aligned, 'D_aligned', opt.which_epoch, opt)
                netD_unaligned = util.load_network(netD_unaligned, 'D_unaligned', opt.which_epoch, opt)

        return netG_for_CT, netD_aligned, netG_for_MR, netD_unaligned

    def generate_synCT(self, MR):
        synCT = self.netG_for_CT(MR)
        return synCT
    
    def generate_synMR(self, CT):
        synMR = self.netG_for_MR(CT)
        return synMR
    
    def compute_generator_loss(self, CT, MR):
        G_losses = {}
        synCT = self.generate_synCT(MR)
        synMR = self.generate_synMR(CT)
            
        unaligned_syn, _, aligned_syn, _ = self.discriminate(CT, MR, synCT, synMR)
        G_losses['GAN_unalignedSyn'] = self.criterionGAN(unaligned_syn, True, for_D=False) * self.opt.lambda_gan
        G_losses['GAN_alignedSyn'] = self.criterionGAN(aligned_syn, True, for_D=False) * self.opt.lambda_gan
        
        if not self.opt.no_MI_loss:
            G_losses['MIc'] = self.criterionMI(synCT, CT) * self.opt.lambda_MI
        if not self.opt.no_L1_loss:
            G_losses['L1c'] = self.criterionL1(synCT, CT) * self.opt.lambda_L1
        if not self.opt.no_vgg_loss:
            G_losses['VGGc'] = self.criterionVGG(synCT.repeat(1,3,1,1), CT.repeat(1,3,1,1)) * self.opt.lambda_vgg
            G_losses['VGGm'] = self.criterionVGG(synMR.repeat(1,3,1,1), CT.repeat(1,3,1,1)) * self.opt.lambda_vgg
        if not self.opt.no_cycle_loss:
            cycCT = self.generate_synCT(synMR)
            cycMR =self.generate_synMR(synCT)
            G_losses['CYCc'] = self.criterionCYC(cycCT, CT) * self.opt.lambda_cyc
            G_losses['CYCm'] = self.criterionCYC(cycMR, MR) * self.opt.lambda_cyc
        if not self.opt.no_ssim_loss:
            G_losses['SSIMc'] = self.criterionSSIM(synCT, CT) * self.opt.lambda_ssim
        if not self.opt.no_GD_loss:
            G_losses['GDc'] = self.criterionGD(synCT, CT) * self.opt.lambda_gdl
            
        return G_losses, {'synCT':synCT,'synMR':synMR,'cycCT':cycCT,'cycMR':cycMR}
    
    def discriminate(self, CT, MR, synCT, synMR):
        syn_concat = torch.cat([CT, synMR], dim=1)
        real_concat = torch.cat([CT, MR], dim=1)
        syn_and_real = torch.cat([syn_concat, real_concat], dim=0)
        discriminator_out = self.netD_aligned(fake_and_real)
        aligned_syn, aligned_real = self.divide_pred(discriminator_out)
        
        syn_and_real = torch.cat([synCT, CT], dim=0)
        discriminator_out = self.netD_unaligned(syn_and_real)
        unaligned_syn, unaligned_real = self.divide_pred(discriminator_out)
        return unaligned_syn, unaligned_real, aligned_syn, aligned_real


    def compute_discriminator_loss(self, CT, MR):

        with torch.no_grad():
            synCT = self.generate_synCT(MR)
            synCT = synCT.detach()
            synCT.requires_grad_()
            synMR = self.generate_synMR(CT)
            synMR = synMR.detach()
            synMR.requires_grad_()
        unaligned_syn, unaligned_real, aligned_syn, aligned_real = self.discriminate(CT, MR, synCT, synMR)

        D_losses['D_unalignedSyn'] = self.criterionGAN(unaligned_syn, False, for_D=True) * self.opt.lambda_gan
        D_losses['D_unalignedReal'] = self.criterionGAN(unaligned_real, True, for_D=True) * self.opt.lambda_gan
        D_losses['D_alignedSyn'] = self.criterionGAN(aligned_syn, False, for_D=True) * self.opt.lambda_gan
        D_losses['D_alignedReal'] = self.criterionGAN(aligned_real, True, for_D=True) * self.opt.lambda_gan
        return D_losses
    
    # Take the prediction of fake and real images from the combined batch
    def divide_preds(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if isinstance(pred, list):
            syn, real = [], []
            for p in pred:
                syn.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            syn = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return syn, real

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0