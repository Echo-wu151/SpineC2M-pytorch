from models.model import Model
from torch.nn import DataParallel

class Trainer():
    def __init__(self, opt):
        self.opt = opt
        self.model = Model(opt)
        if len(opt.gpu_ids) > 0:
            self.model = DataParallel(self.model, device_ids=opt.gpu_ids)
            self.model_on_one_gpu = self.model.module
        else:
            self.model_on_one_gpu = self.model

        self.synMR = None
        self.cycMR = None
        self.synCT = None
        self.cycCT = None
        self.g_losses = {}
        self.d_losses = {}
        if opt.isTrain:
            self.optimizerG, self.optimizerD = \
                self.model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

    def run_generator_one_step(self, CT, MR):
        
        self.optimizerG.zero_grad()
        g_losses, generated = self.model(CT, MR, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizerG.step()
        self.g_losses = g_losses

    def run_discriminator_one_step(self, CT, MR):

        self.optimizerD.zero_grad()
        d_losses = self.model(CT, MR, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizerD.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.model_on_one_gpu.save(epoch)

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizerD.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizerG.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
