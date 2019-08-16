from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.vgg import VGG19relu
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


class VGGLoss(nn.Module):
    """
    Perceptual loss that uses a pretrained VGG network
    """
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.vgg = VGG19relu().cuda(gpu_ids[0])
        self.opt = opt
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i, (x_ft, y_ft) in enumerate(zip(x_vgg, y_vgg)):
            loss += self.weights[i] * self.criterion(x_ft, y_ft.detach())
        return loss


    
class GradientDifferenceLoss(nn.Module):
    def forward(self, preds, target):
        pdx, pdy = preds[:, :, 1:, :] - preds[:, :, :-1, :], preds[:, :, :, 1:] - preds[:, :, :, :-1]
        tdx, tdy = target[:, :, 1:, :] - target[:, :, :-1, :], target[:, :, :, 1:] - target[:, :, :, :-1]
        
        GD = torch.stack([(tdx-pdx).abs(), (tdy-pdy).abs()])
        return GD.mean()
