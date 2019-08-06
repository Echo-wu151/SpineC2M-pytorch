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

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()
            if opt.plus_L1_loss:
                self.criterionL1 = torch.nn.L1Loss()
            if opt.plus_L2_loss:
                self.criterionL2 = torch.nn.MSELoss()
            if opt.plus_MI_loss:
                self.criterionMI = networks.MILoss()
            if opt.plus_BCE_loss:
                self.criterionBCE = torch.nn.BCELoss()
            if opt.plus_TV_loss:
                self.criterionTV = networks.TVLoss()
            self.beta = torch.distributions.beta.Beta(0.5, 0.5)

        self.count = 0.
        self.eps = 0.
    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):

        input_semantics, real_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image)
            return g_loss, generated

        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image)
            return d_loss

        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, real_image)
            return fake_image

        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):

        G_params = list(self.netG.parameters())
        E_params = list(self.netE.parameters()) if opt.use_vae else None
        D_params = list(self.netD.parameters()) if opt.isTrain else None

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2)) if opt.isTrain else None

        return optimizer_G, optimizer_D

    
    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)

    ##########################################################################
    # Private helper methods
    ##########################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netG, netD

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):

        if self.use_gpu():
            keys = ("image", "label", "instance")
            for k, d in data.items():
                if k in keys:
                    data[k] = d.cuda()

        if not self.opt.no_label:
            # move to GPU and change data types
            data['label'] = data['label'].long()
            # create one-hot label map
            label_map = data['label']
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            input_label = self.FloatTensor(bs, nc, h, w).zero_()
            input_semantics = input_label.scatter_(1, label_map, 1.0)
        else:
            input_semantics = data['label']

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat(
                (input_semantics, instance_edge_map), dim=1)

        return input_semantics, data['image']

    def compute_generator_loss(self, input_semantics, real_image):
        # for_D : criterion for discriminator
        G_losses = {}

        fake_image, KLD_loss = self.generate_fake(
            input_semantics, real_image)

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_D=False) * self.opt.lambda_gan
        lamb = self.beta.sample().item()

        pred = [list(map(lambda x, y: x * lamb + y * (1 - lamb), r, f))
                for r, f in zip(pred_real, pred_fake)]
        G_losses['G_mix'] = self.criterionGAN(pred, True,
                                              for_D=False) * self.opt.lambda_gan
        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(
                        num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if self.opt.plus_MI_loss:
            G_losses['MI'] = self.criterionMI(fake_image, real_image) \
                * self.opt.lambda_MI

        if self.opt.plus_BCE_loss:
            G_losses['BCE'] = self.criterionBCE(fake_image.add(1).div(2), real_image.add(1).div(2)) \
                * self.opt.lambda_BCE

        if self.opt.plus_L1_loss:
            G_losses['L1'] = self.criterionL1(fake_image, real_image) \
                * self.opt.lambda_L1
        if self.opt.plus_L2_loss:
            G_losses['L2'] = self.criterionL2(fake_image, real_image) \
                * self.opt.lambda_L2
        if self.opt.plus_TV_loss:
            G_losses['TV'] = self.criterionTV(fake_image) \
            * self.opt.lambda_TV

        size = list(fake_image.size())
        if (int(size[1]) != 3) and (not self.opt.no_vgg_loss):
            size[1] = 3
            fake_image = fake_image.expand(*size)
            real_image = real_image.expand(*size)

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_encoder_loss(self, input_semantics, real_image):
        G_losses = {}

        fake_image = self.generate_fake(
            input_semantics, real_image,
            is_backward_z=True)

        if self.opt.plus_BCE_loss:
            E_losses['BCE'] = self.criterionBCE(fake_image.add(1).div(2), real_image.add(1).div(2)) \
                * self.opt.lambda_BCE

        if self.opt.plus_L1_loss:
            E_losses['L1'] = self.criterionL1(fake_image, real_image) \
                * self.opt.lambda_L1
        if self.opt.plus_L2_loss:
            E_losses['L2'] = self.criterionL2(fake_image, real_image) \
                * self.opt.lambda_L2

        return E_losses

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_D=True) * self.opt.lambda_gan
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_D=True) * self.opt.lambda_gan
        lamb = self.beta.sample()

        pred = [list(map(lambda x, y: x * lamb + y * (1 - lamb), r, f))
                for r, f in zip(pred_real, pred_fake)]

        D_losses['D_mix'] = self.criterionGAN(pred, False,
                                              for_D=True) * self.opt.lambda_gan

        return D_losses

    def encode_z(self, real_image):

        mu, logvar = self.netE(real_image)

        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image,
                      is_backward_z=False):

        fake_image = self.netG(input_semantics)

        return fake_image

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out = self.netD(fake_and_real)
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.count % self.opt.z_duration == 0:
            self.eps = torch.randn_like(std)
        self.count += 1
        return self.eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0