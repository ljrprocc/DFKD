# from sre_parse import _OpGroupRefExistsType
from distutils.command.config import LANG_EXT
from torch import nn
import torch
import torch.nn.functional as F
from torchvision import transforms
import math

from .base import BaseSynthesis
import datafree
from datafree.hooks import DeepInversionHook
from datafree.utils import ImagePool, DataIter, clip_images
from datafree.criterions import jsdiv, kldiv
from kornia import augmentation
import numpy as np
from datafree.models.rl import Environment
from datafree.models.rl import Actor, Critic

def difficulty_loss(anchor, teacher, t_out, logit_t, ds='cifar10', hard_factor=0., tau=10, device='cpu'):
    batch_size = anchor.size(0)
    with torch.no_grad():
        pseudo_label, anchor_t_out = teacher(anchor.to(device).detach(), return_features=True)
        pseudo_label = pseudo_label.argmax(1)
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
        d = torch.mm(anchor_t_out, t_out.T)
        N_an, N_batch = d.size()
        # positive_negative_border = torch.quantile(d, q=0.1, dim=1)
        # d_pos = d[:, d <= positive_negative_border]
        # d_neg = d[:, d > positive_negative_border]
        sorted_d, indice_d = torch.sort(d, dim=1)
        d_pos = sorted_d[:, :int(0.1 * N_batch)]
        d_neg = sorted_d[:, int(0.1 * N_batch):]
        # Get positive DA index
        p_pos = torch.softmax(d_pos / tau, dim=1)
        p_da_pos = torch.quantile(p_pos, q=hard_factor, dim=1).unsqueeze(1)
        pos_loss = torch.sum(p_pos * torch.log(p_pos / p_da_pos), dim=1).mean().abs()
        # Get Negative DA index
        p_neg = torch.softmax(d_neg / tau, dim=1)
        p_da_neg = torch.quantile(p_neg, q=1-hard_factor, dim=1).unsqueeze(1)
        neg_loss = torch.sum(p_neg * torch.log(p_neg / p_da_neg), dim=1).mean().abs()

        # print(pos_loss, neg_loss)
        return pos_loss+neg_loss, pos_loss, neg_loss

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
    def __init__(self, teacher, student, G_list, num_classes, img_size, nz, iterations=100, lr_g=0.1, synthesis_batch_size=128, sample_batch_size=128, save_dir='run/improved_cudfkd', transform=None, normalizer=None, device='cpu', use_fp16=False, distributed=False, lmda_ent=0.5, adv=0.10, oh=0, act=0, l1=0.01, mk=0.2, depth=2, adv_type='js', bn=0, T=5, memory=False, evaluator=None, gk=0.9, tau=10, hard=1.0):
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
        self.data_pool = ImagePool(root=self.save_dir, save=False)
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
        self.mk = mk
        self.k = 1
        self.gk = gk
        self.tau = tau
        self.hard = hard
        # Reinforcement settings
        # self.rl_iters = rl_iters
        # self.actor = Actor(state_size=self.teacher.linear.in_features+self.student.linear.in_features, action_size=1).to(device)
        # self.critic = Critic(state_size=self.teacher.linear.in_features+self.student.linear.in_features, action_size=synthesis_batch_size).to(device)
        # self.env = Environment(models=(self.teacher, self.student, G_list[0]), evaluator=evaluator, batch_size=synthesis_batch_size, latent_dim=nz, device=device)
        # self.aug = transforms.Compose([
        #     augmentation.RandomCrop(size=[img_size, img_size], padding=4),
        #     augmentation.RandomHorizontalFlip(),
        #     normalizer,
        # ])
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
        # z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device).requires_grad_()
        # Analysis this framework.
        # reset_model(G)
        # optimizer = torch.optim.Adam([{'params': G.parameters()}, {'params': [z]}], self.lr_g, betas=[0.5, 0.999])
        for i in range(self.iterations[l]):
            # if i % 50 == 0:
            #     print(i)
            
            z = torch.randn(self.synthesis_batch_size, self.nz).to(self.device)
           
            G.train()
            # optimizer.zero_grad()
            self.optimizers[l].zero_grad()
            
            # Rec and variance
            # mu_theta, logvar_theta = G(z1, l=l)
            mu_theta = G(z, l=l)
            samples = self.normalizer(mu_theta)
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
            if not warmup and self.memory:
                anchor = self.data_iter.next()
                loss_hard, loss_pos, loss_neg = difficulty_loss(anchor, self.teacher, t_feat, logit_t=t_out, hard_factor=hard_factor, tau=self.tau, device=self.device)
                loss = self.lmda_ent * ent + self.adv * loss_adv+ self.oh * loss_oh + self.act * loss_act + self.bn * loss_bn + self.hard * loss_hard
            else:
                loss = self.lmda_ent * ent + self.adv * loss_adv+ self.oh * loss_oh + self.act * loss_act + self.bn * loss_bn
            
            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = mu_theta
                    # print(best_inputs.max(), best_inputs.min())
                    
           
            loss.backward()
            self.optimizers[l].step()
            # self.x = best_inputs
            # optimizer.step()
            # print(best_inputs.shape)

        if self.memory and warmup:
            self.update_loader(best_inputs=best_inputs)
           
        # exit(-1)
        # self.student.train()
        return {'synthetic': best_inputs}

    @torch.no_grad()
    def sample(self, l=0, history=False, warmup=True):
        if not history:
            self.G_list[l].eval() 
            z = torch.randn( size=(self.sample_batch_size, self.nz), device=self.device )
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,), device=self.device)
            inputs = self.G_list[l](z, l=l)
            # print(inputs)
            # exit(-1)
            if not warmup:
                self.k = max(self.k * (1 - self.gk), self.mk)
                # # Choose top-k ratio subset and fill out for length self.synthesis_bs
                # all_repeats = math.ceil(1 / self.k)
                t_out = self.teacher(self.normalizer(inputs))
                s_out = self.student(self.normalizer(inputs))
                loss = datafree.criterions.kldiv(s_out, t_out, T=self.T, reduction='none').sum(1)
                _, indice = torch.sort(loss)
                selected_index = indice[-int(self.k * self.synthesis_batch_size):]
                inputs = inputs[selected_index]
                # print(inputs.shape)

        else:
            inputs = self.data_iter.next()
        return inputs

    def update_loader(self, best_inputs):
        self.data_pool.add(best_inputs)
        dst = self.data_pool.get_dataset(transform=self.transform)
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dst)
        else:
            train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)