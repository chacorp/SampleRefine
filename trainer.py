"""
Reference:
    Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
    Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch

class Trainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """
    def __init__(self, opt=None, device='cpu', train_log=None):
        self.opt = opt
        self.train_log=train_log
        self.device = device

        from sample_refine import Model

        self.model = Model(opt, self.device)

        self.model_on_one_gpu = self.model

        if opt.mode in ['train', 'debug'] :
            self.optimizer_G, self.optimizer_D = self.model_on_one_gpu.create_optimizers()
        
        self.old_lr = opt.lr

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses = self.model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses

    def run_discriminator_one_step(self, data):
        if self.opt.G in ['S1','S2']:
            self.d_losses = {'no D loss':torch.zeros([1]).to(self.device)}
            return
        self.optimizer_D.zero_grad()
        d_losses = self.model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_validation_result(self, data):
        return self.model(data, mode='validation')

    def get_latest_losses(self):
        return self.g_losses, self.d_losses

    def save_model(self, path, epoch):
        self.model_on_one_gpu.save(path, epoch)

    def update_progress(self, manual=None):
        old_scale = self.model.tps_scale

        if manual != None:
            self.model.tps_scale = manual
        else:
            if self.model.tps_scale == 0:
                self.model.tps_scale = 0.1
            else:
                self.model.tps_scale = min(old_scale + self.opt.pg_weight, 0.25) # maximum 16 pixel for 256 x 256

        self.train_log.info('update transform scale: {:.4f} -> {:.4f}'.format(old_scale, self.model.tps_scale))

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            new_lr_G = new_lr
            new_lr_D = new_lr

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G

            self.train_log.info('update learning rate: %f -> %f' % (self.old_lr, new_lr))            
            self.old_lr = new_lr
