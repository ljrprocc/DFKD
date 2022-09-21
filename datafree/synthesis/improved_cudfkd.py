# from sre_parse import _OpGroupRefExistsType
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
    def __init__(self, teacher, student, G_list, num_classes, img_size, nz, iterations=100, lr_g=0.1, synthesis_batch_size=128, sample_batch_size=128, save_dir='run/improved_cudfkd', transform=None, normalizer=None, device='cpu', use_fp16=False, distributed=False, lmda_ent=0.5, adv=0.10, oh=0, act=0, l1=0.01, mk=0.2, depth=2, adv_type='js', bn=0, T=5, memory=False, strategy='MH', gk=0.9):
        super(MHDFKDSynthesizer, self).__init__(teacher, student)
        self.save_dir = save_dir
        self.img_size = img_size 
        self.iterations = iterations
        self.lr_g = lr_g
        self.normalizer = normalizer
        self.strategy = strategy
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
        self.aug = transforms.Compose([
            augmentation.RandomCrop(size=[img_size, img_size], padding=4),
            augmentation.RandomHorizontalFlip(),
            normalizer,
        ])
        for i, G in enumerate(self.G_list):
            reset_model(G)
            optimizer = torch.optim.Adam(G.parameters(), self.lr_g, betas=[0.9, 0.99])
            self.optimizers.append(optimizer)

        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m))

        # assert len(self.hooks)>0, 'input model should contains at least one BN layer for DeepInversion and Probablistic KD.'

    def synthesize(self, l=0, gv=None):
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
            
            loss = self.lmda_ent * ent + self.adv * loss_adv+ self.oh * loss_oh + self.act * loss_act + self.bn * loss_bn
            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = mu_theta
                    # print(best_inputs.max(), best_inputs.min())
                    
           
            loss.backward()
            self.optimizers[l].step()
            # optimizer.step()
           
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
                samples = []
                # all_samples = torch.randn(0, )
                # i = 0
                t_out = self.teacher(self.normalizer(inputs))
                s_out = self.student(self.normalizer(inputs))
                loss = datafree.criterions.kldiv(s_out, t_out, T=self.T, reduction='none').sum(1)
                _, indice = torch.sort(loss)
                
                #     samples.append(inputs[indice[-int(self.k * self.synthesis_batch_size):]].cpu())
                #     all_samples = torch.cat(samples, 0)
                #     if i < all_repeats - 1:
                #         shape0 = math.ceil((self.sample_batch_size - all_samples.size(0)) / self.k)
                #         # print(shape0)
                #         z = torch.randn( size=(shape0, self.nz), device=self.device )
                #         inputs = self.G_list[l](z, l=l)
                #     i += 1
                # all_samples = torch.cat(samples, 0)
                # random_index = torch.randint(0, all_samples.size(0), (self.synthesis_batch_size, ))
                # # print(inputs[indice])
                # inputs = all_samples[random_index].to(self.device)
                # print(all_samples.size(0))
                # exit(-1)
                # print(inputs.mean(), inputs.max(), inputs.min())
                # Choose top-k ratio subset
                # random_index = torch.randint(0, self.synthesis_batch_size, (int(self.synthesis_batch_size * self.k), ))
                # inputs = inputs[random_index]
                selected_index = indice[-int(self.k * self.synthesis_batch_size):]
                samples.append(inputs[selected_index].cpu())
                all_samples = torch.cat(samples, 0)
                while all_samples.size(0) < self.synthesis_batch_size:
                    new_samples = self.aug(all_samples)
                    samples.append(new_samples)
                    all_samples = torch.cat(samples, 0)
                # all_samples = torch.cat(samples, 0)
                random_index = np.random.choice(all_samples.size(0), self.synthesis_batch_size, replace=False)
                random_index = torch.LongTensor(random_index)
                inputs = all_samples[random_index].to(self.device)
                # print(inputs.mean())
                # exit(-1)
                # inputs = inputs[selected_index]
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