from distutils.command.config import LANG_EXT
from torch import nn
import torch
import torch.nn.functional as F
from torchvision import transforms
import math
import os

from .base import BaseSynthesis
import datafree
from datafree.hooks import DeepInversionHook
from datafree.utils import ImagePool, DataIter, clip_images, FeaturePool
from datafree.criterions import jsdiv, kldiv
from kornia import augmentation
import numpy as np

def difficulty_loss(anchor, teacher, t_out, logit_t, ds='cifar10', hard_factor=0., tau=10, device='cpu'):
    batch_size = anchor.size(0)
    with torch.no_grad():
        # t_logit, anchor_t_out = teacher(anchor.to(device).detach(), return_features=True)
        t_logit = teacher(anchor.to(device).detach())
        anchor_t_out = anchor.to(device)
        # pseudo_label = pseudo_label.argmax(1)
    # loss = 0.
    pos_loss = 0.
    neg_loss = 0.
    if ds == 'cifar10':
        # for i in range(batch_size):
        #     this_anchor = anchor_t_out[i].unsqueeze(0)
        #     pos_features = t_out[pseudo_label[i] == logit_t.argmax(1)]
        #     neg_features = t_out[pseudo_label[i] != logit_t.argmax(1)]
        #     d_pos = torch.mm(this_anchor, pos_features.T)
        #     d_neg = torch.mm(this_anchor, neg_features.T)
        #     # Get positive DA index
        #     p_pos = torch.softmax(d_pos / tau, dim=1)
        #     p_da_pos = torch.quantile(p_pos, q=hard_factor, dim=1).item()
        #     l_pos = torch.sum(p_pos * torch.log(p_pos / p_da_pos))
        #     # Get Negative DA index
        #     p_neg = torch.softmax(d_neg / tau, dim=1)
        #     p_da_neg = torch.quantile(p_neg, q=1-hard_factor, dim=1).item()
        #     l_neg = torch.sum(p_neg * torch.log(p_neg / p_da_neg))
        #     pos_loss += l_pos
        #     neg_loss += l_neg
        normalized_anchor_t_out, normalized_t_out = F.normalize(anchor_t_out, dim=1), F.normalize(t_out, dim=1)
        d = torch.mm(normalized_anchor_t_out, normalized_t_out.T)
        N_an, N_batch = d.size()
        # positive_negative_border = torch.quantile(d, q=0.1, dim=1)
        # d_pos = d[:, d <= positive_negative_border]
        # d_neg = d[:, d > positive_negative_border]
        
        sorted_d, indice_d = torch.sort(d, dim=1)
        d_pos = sorted_d[:, -int(0.1 * N_batch):]
        d_neg = sorted_d[:, :-int(0.1 * N_batch)]
        d_mask = torch.zeros_like(indice_d)
        d_mask = d_mask.scatter(1, indice_d[:, -int(0.1*N_batch):], 1)
        p_t_anchor = torch.softmax(t_logit, 1)
        p_t_batch = torch.softmax(logit_t, 1)
        kld_matrix = -torch.mm(p_t_anchor, p_t_batch.T.log()) + torch.diag(torch.mm(p_t_anchor, p_t_anchor.T.log())).unsqueeze(1)
        l_kld = ((kld_matrix * d_mask).sum(1) / d_mask.sum(1)).mean()
        # Get positive DA index
        p_pos = torch.softmax(d_pos / tau, dim=1)
        p_da_pos = torch.quantile(p_pos, q=1-hard_factor, dim=1).unsqueeze(1)
        pos_loss = torch.sum(p_pos * torch.log(p_pos / p_da_pos).abs(), dim=1).mean()
        # Get Negative DA index
        p_neg = torch.softmax(d_neg / tau, dim=1)
        p_da_neg = torch.quantile(p_neg, q=hard_factor, dim=1).unsqueeze(1)
        neg_loss = torch.sum(p_neg * torch.log(p_neg / p_da_neg).abs(), dim=1).sum()

        # print(pos_loss, neg_loss)
        return pos_loss, pos_loss, neg_loss, l_kld

def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class MHDFKDSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, G_list, num_classes, img_size, nz, iterations=100, lr_g=0.1, synthesis_batch_size=128, sample_batch_size=128, save_dir='run/improved_cudfkd', transform=None, normalizer=None, device='cpu', use_fp16=False, distributed=False, lmda_ent=0.5, adv=0.10, oh=0, act=0, l1=0.01, depth=2, adv_type='js', bn=0, T=5, memory=False, evaluator=None, tau=10, hard=1.0, mu=0.5, bank_size=10, mode='memory', kld=0.1):
        super(MHDFKDSynthesizer, self).__init__(teacher, student)
        self.save_dir = save_dir
        self.img_size = img_size 
        self.iterations = iterations
        self.lr_g = lr_g
        self.normalizer = normalizer
        self.evaluator = evaluator
        # self.strategy = strategy
        # Avoid duplicated saving.
        # if os.path.exists(self.save_dir):
        #     shutil.rmtree(self.save_dir)
        # self.data_pool = ImagePool(root=self.save_dir, save=False)
        self.data_pool = FeaturePool(root=self.save_dir)
        self.data_iter = None
        self.transform = transform
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.use_fp16 =use_fp16
        self.device = device
        self.num_classes = num_classes
        self.G_list = G_list
        # self.E_list = E_list
        self.adv_type = adv_type
        self.lmda_ent = lmda_ent
        self.adv = adv
        self.nz = nz
        self.L = depth + 1
        self.l1 = l1
        self.bn = bn
        self.distributed = distributed
        
        self.oh = oh
        self.act = act
        self.T = T
        self.memory = memory
        # self._get_teacher_bn()
        self.optimizers = []
        self.tau = tau
        self.hard = hard
        self.mu = mu
        self.mode = mode
        self.kld = kld
        if not os.path.exists(os.path.join(self.save_dir, 'buffer.pt')):
            self.anchor_bank = torch.randn(bank_size, synthesis_batch_size, teacher.linear.in_features)
        else:
            self.anchor_bank = torch.load(os.path.join(self.save_dir, 'buffer.pt'))
        for i, G in enumerate(self.G_list):
            reset_model(G)
            optimizer = torch.optim.Adam(G.parameters(), self.lr_g, betas=[0.9, 0.99])
            self.optimizers.append(optimizer)

        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m))


    def synthesize(self, l=0, gv=None, hard_factor=0., warmup=False):
        self.student.eval()
        self.teacher.eval()
        # optimizers = []
        G = self.G_list[l]
        best_cost = 9999999
        best_inputs = None
        if gv is not None:
            g, v= gv
            v = v.to(self.device)
            g = g.to(self.device)
        for i in range(self.iterations[l]):
            # if i % 50 == 0:
            #     print(i)
            
            z = torch.randn(self.synthesis_batch_size, self.nz).to(self.device)
            # print(z)
            G.train()
            # optimizer.zero_grad()
            self.optimizers[l].zero_grad()
            
            mu_theta = G(z, l=l)
            samples = self.normalizer(mu_theta)
            # print(samples)
            x_inputs = self.normalizer(samples, reverse=True)
            t_out, t_feat = self.teacher(samples, l=l, return_features=True)
            # print(t_out)
            p = F.softmax(t_out / self.T, dim=1).mean(0)
            ent = -(p*p.log()).sum()
            # if targets is None:
            loss_oh = F.cross_entropy( t_out, t_out.max(1)[1])
            # loss_oh = F.cross_entropy( t_out, targets )
            loss_act = - t_feat.abs().mean()
            if self.bn > 0:
                loss_bn = sum([h.r_feature for h in self.hooks])
            else:
                loss_bn = torch.zeros(1).to(self.device)
            
            # Negative Divergence.
            if self.adv > 0:
                s_out = self.student(samples, l=l)
                if self.adv_type == 'js':
                    l_js = jsdiv(s_out, t_out, T=3)
                    loss_adv = 1.0-torch.clamp(l_js, 0.0, 1.0)
                if self.adv_type == 'kl':
                    mask = (s_out.max(1)[1]==t_out.max(1)[1]).float()
                    loss_adv = -(kldiv(s_out, t_out, reduction='none', T=3).sum(1) * mask).mean()
            else:
                loss_adv = torch.zeros(1).to(self.device)
            
            # After Warmup, should include the following positive-negative pairs objectives.
            # Anchor sampling:
            # warmup = True
            if not warmup:
                if self.mode == 'memory':
                    if self.data_iter is None and len(self.data_pool.datas) > 0:
                        dst = self.data_pool.get_dataset(transform=self.transform)
                        if self.distributed:
                            train_sampler = torch.utils.data.distributed.DistributedSampler(dst)
                        else:
                            train_sampler = None
                        loader = torch.utils.data.DataLoader(
                            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
                            num_workers=4, pin_memory=True, sampler=train_sampler)
                        self.data_iter = DataIter(loader)
                    anchor = self.data_iter.next()
                else:
                    import random
                    random_index = random.randint(0, self.anchor_bank.size(0) - 1)
                    anchor = self.anchor_bank[random_index]
                # loss_hard, loss_pos, loss_neg, loss_kld = difficulty_loss(anchor, self.teacher, t_feat, logit_t=t_out, hard_factor=hard_factor, tau=self.tau, device=self.device)
                loss_hard, loss_pos, loss_neg, loss_kld = difficulty_loss(anchor, self.teacher.linear, t_feat, logit_t=t_out, hard_factor=hard_factor, tau=self.tau, device=self.device)
                loss = self.lmda_ent * ent + self.adv * loss_adv+ self.oh * loss_oh + self.act * loss_act + self.bn * loss_bn + self.hard * loss_hard + self.kld * loss_kld
            else:
                loss = self.lmda_ent * ent + self.adv * loss_adv+ self.oh * loss_oh + self.act * loss_act + self.bn * loss_bn
            
            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = mu_theta
                    # print(best_inputs)
                    
           
            loss.backward()
            self.optimizers[l].step()
            # self.x = best_inputs
            # optimizer.step()
            # print(best_inputs.shape)

        if self.memory or warmup:
            # self.update_loader(best_inputs=best_inputs)
            self.update_loader(best_inputs=t_feat)
           
        # exit(-1)
        # self.student.train()
        return {'synthetic': best_inputs}

    @torch.no_grad()
    def sample(self, l=0, warmup=True):
        # Formal Mode
        # if not self.memory or not warmup:
        # Global memory:
        if not self.memory:
            # print('***********')
            self.G_list[l].eval() 
            z = torch.randn( size=(self.sample_batch_size, self.nz), device=self.device )
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,), device=self.device)
            inputs = self.G_list[l](z, l=l)
        else:
            inputs = self.data_iter.next()
            # print(inputs)
            # inputs = self.normalizer(inputs)
            # print(inputs)
        return inputs

    def update_loader(self, best_inputs):
        if self.mode == 'memory':
            self.data_pool.add(best_inputs)
            # dst = self.data_pool.get_dataset(transform=self.transform)
            dst = self.data_pool.get_dataset()
            if self.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dst)
            else:
                train_sampler = None
            loader = torch.utils.data.DataLoader(
                dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
                num_workers=4, pin_memory=True, sampler=train_sampler)
            self.data_iter = DataIter(loader)
        else:
            import random
            bank_index = random.randint(0, self.anchor_bank.size(0) - 1)
            
            self.anchor_bank[bank_index] = best_inputs.detach().cpu() * self.mu + self.anchor_bank[bank_index] * (1 - self.mu)
            self.data_pool.add(self.anchor_bank, replace=True)

