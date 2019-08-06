from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.vgg import VGG19
import numpy as np


class GANLoss(nn.Module):
    def __init__(self, gan_mode, tensor=torch.Tensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = opt.target_real_label
        self.fake_label = opt.target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.tensor = tensor
        self.gan_mode = gan_mode
        if gan_mode == 'ls': pass
        elif gan_mode == 'original': pass
        elif gan_mode == 'w': pass
        elif gan_mode == 'hinge': pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, tensor, target_is_real):
        """
        get real/fake label_tensor
        """
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(tensor)

        if self.fake_label_tensor is None:
            self.fake_label_tensor = self.tensor(1).fill_(self.fake_label)
            self.fake_label_tensor.requires_grad_(False)
        return self.fake_label_tensor.expand_as(tensor)

    def get_zero_tensor(self, tensor):
        """
        get fake label tensor
        """
        if self.zero_tensor is None:
            self.zero_tensor = self.tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(tensor)

    def loss(self, tensor, target_is_real, for_discriminator=True):
        """
        choose gan_mode
        """
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(tensor, target_is_real)
            loss = F.binary_cross_entropy_with_logits(tensor, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(tensor, target_is_real)
            return F.mse_loss(tensor, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = (tensor - 1).min(self.get_zero_tensor(tensor))
                    loss = -minval.mean()
                else:
                    minval = (-tensor - 1).min(self.get_zero_tensor(tensor))
                    loss = -minval.mean()
            else:
                assert target_is_real, "The G's hinge loss must be aiming for real"
                loss = -tensor.mean()
            return loss
        # wgan
        if target_is_real:
            return -tensor.mean()
        return tensor.mean()

    def __call__(self, tensorlist, target_is_real, for_D=True):

        if isinstance(tensorlist, list):
            loss = 0
            for tensor in tensorlist:
                if isinstance(tensor, list):
                    tensor = tensor[-1]
                loss_tensor = self.loss(tensor, target_is_real, for_D)
                bsize = loss_tensor.size(0) if loss_tensor.dim() else 1
                new_loss = loss_tensor.view(bsize, -1).mean(dim=1)
                loss += new_loss
            return loss / len(tensorlist)
        return self.loss(tensorlist, target_is_real, for_D)
    
    
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(BATCH_SIZE, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda() if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if use_cuda:
            interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                      disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty



class VGGLoss(nn.Module):
    """
    Perceptual loss that uses a pretrained VGG network
    """
    def __init__(self, gpu_ids, vgg_mode='pool'):
        super(VGGLoss, self).__init__()
        self.vgg_mode = vgg_mode
        self.criterion = nn.L1Loss()
        self.vgg = VGG19().cuda(gpu_ids[0])
        self.opt = opt
        if vgg_mode == 'pool':
            self.weights = [1.0] * 3
        else:
            self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i, (x_ft, y_ft) in enumerate(zip(x_vgg, y_vgg)):
            loss += self.weights[i] * self.criterion(x_ft, y_ft.detach())
            if not self.opt.no_style_loss:
                x_gram, y_gram = self.gram_matrix(x_ft), self.gram_matrix(y_ft)
                loss += self.opt.lambda_style * self.weights[i] * self.criterion(x_gram, y_gram.detach())
        return loss
    
    def gram_matrix(self, input):
        a, b, c, d = input.size()  
        feats = input.view(a * b, c * d)
        G = torch.mm(feats, feats.t()) 

        return G.div(a * b * c * d)


class MILoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.gaugram = GaussianHistogram(bins=1500, min=-1,max=1,sigma=1)
    def forward(self, x, y):
        assert x.size() == y.size(), "input x and y should be same dim"
        bsize = x.size(0)
        pxy = []
        for i in range(bsize):
            xi = x[i]
            yi = y[i]
            xi = self.gaugram(xi.view(-1))[:, None].float()
            yi = self.gaugram(yi.view(-1))[None, :].float()
            gaugram = torch.mm(xi, yi)
            pxy.append(gaugram / gaugram.sum())
        pxy = torch.stack(pxy, dim=0)
        joint_hgram = -pxy * (pxy + 1e-6).log()
        return joint_hgram
    
class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1).cuda()
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        x = x.sum(dim=1)
        return x

    