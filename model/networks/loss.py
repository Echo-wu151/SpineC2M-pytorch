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

    
    
    
def calc_gradient_penalty(netD, real_data, fake_data):
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
    
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(
            pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    
class KLDLoss(nn.Module):
    """
    KL Divergence loss used in VAE with an image encoder
    """

    def forward(self, mu, logvar):
        size = mu.size()
        mu = mu.view(size[0], -1)
        logvar = logvar.view(size[0], -1)

        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()

    
from numpy import flip
import numpy as np
from scipy.signal import convolve2d, correlate2d
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class MIFunction(Function):
    @staticmethod
    def forward(ctx, x, y):
        # detach so we can cast to NumPy
        x, y = x.detach(), y.detach() 
        histx = ndimage.measurements.historgram(x.numpy(), -1, 1, 1000)
        histy = ndimage.measurements.historgram(y.numpy(), -1, 1, 1000)
        result = histx - histy
        ctx.save_for_backward(x, y)
        return torch.as_tensor(result, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        x, y = ctx.saved_tensors
        grad_output = grad_output.numpy()
        x = grad_output + y
        y = grad_output + y

        return torch.from_numpy(x),torch.from_numpy(y)


class ScipyConv2d(Module):
    def __init__(self, filter_width, filter_height):
        super(ScipyConv2d, self).__init__()
        self.filter = Parameter(torch.randn(filter_width, filter_height))
        self.bias = Parameter(torch.randn(1, 1))

    def forward(self, input):
        return ScipyConv2dFunction.apply(input, self.filter, self.bias)
    
class MILoss(nn.Module):
    """
    Mutual information loss used
    when paired images(ce-nce) is not alligned
    """
    def __init__(self):
        super().__init__()
        self.gaugram = GaussianHistogram(bins=250, min=-1,max=1,sigma=1)
    def forward(self, x, y):
        assert x.size() == y.size(), "input x and y should be same dim"
        bsize = x.size(0)
        pxy = []
        for i in range(bsize):
            xi=x[i]
            yi=y[i]
            gaugram = self.gaugram(xi.view(-1))[:, None].float().mm(self.gaugram(yi.view(-1))[None, :].float())
            pxy.append(gaugram / gaugram.sum())
        pxy = torch.stack(pxy, dim=0)
        joint_hgram = -pxy * (pxy + 1e-8).log()
        return MIFunction.apply(x, y)
    
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

    
class TVLoss(nn.Module):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (B, 3, H, W) holding an input image.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
    def forward(self, img):

        w_variance = self.criterion(img[:,:,:,:-1], img[:,:,:,1:])
        h_variance = self.criterion(img[:,:,:-1,:], img[:,:,1:,:])
        loss = (h_variance + w_variance)
        return loss
    