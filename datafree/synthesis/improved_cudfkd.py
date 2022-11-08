from torch import nn
import torch
import torch.nn.functional as F
import random
import os
import shutil

from .base import BaseSynthesis
import datafree
from datafree.hooks import DeepInversionHook
from datafree.utils import MoCo, DataIter, FeaturePool
from datafree.criterions import jsdiv, kldiv
from datafree.datasets.utils import curr_v, lambda_scheduler  

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
    def __init__(self, teacher, student, G_list, num_classes, img_size, nz, iterations=100, lr_g=0.1, synthesis_batch_size=128, sample_batch_size=128, save_dir='run/improved_cudfkd', transform=None, normalizer=None, device='cpu', use_fp16=False, distributed=False, lmda_ent=0.5, adv=0.10, oh=0, act=0, l1=0.01, depth=2, adv_type='js', bn=0, T=5, memory=False, evaluator=None, tau=10, hard=1.0, mu=0.5, k=1., mode='memory', neg=0.1, debug=False, n_neg=6144):
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
        # print('********')
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
        self.k = k
        self.distributed = distributed
        # self.neg_bank = Queue(capacity=100)
        if distributed:
            mod = self.teacher.module
        else:
            mod = self.teacher
        
        if hasattr(mod, 'linear'):
            dims = mod.linear.in_features
        elif hasattr(mod, 'fc'):
            dims = mod.fc.in_features
        else:
            dims = mod.classifier.in_features
        
        self.neg_bank = MoCo(dim=dims, K=n_neg, T=tau, device=device, distributed=distributed)
        
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
        self.n_neg = n_neg
        self.neg = neg
        self.debug = debug
        
        # if not os.path.exists(os.path.join(self.save_dir, 'buffer.pt')):
        #     self.anchor_bank = torch.randn(bank_size, synthesis_batch_size, module.in_features)
        # else:
        #     self.anchor_bank = torch.load(os.path.join(self.save_dir, 'buffer.pt'))
        for i, G in enumerate(self.G_list):
            reset_model(G)
            optimizer = torch.optim.Adam(G.parameters(), self.lr_g, betas=[0.9, 0.99])
            self.optimizers.append(optimizer)

        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m))
    
    
    def _set_head(self, teacher, student):
        if hasattr(teacher, 'linear'):
            # ResNet
            self.head = teacher.linear
            # self.stu_head = student.linear
        elif hasattr(teacher, 'fc'):
            self.head = teacher.fc
        elif hasattr(teacher, 'classifier'):
            self.head = teacher.classifier

        if hasattr(student, 'linear'):
            # ResNet
            self.stu_head = student.linear
            # self.stu_head = student.linear
        elif hasattr(student, 'fc'):
            self.stu_head = student.fc
        elif hasattr(student, 'classifier'):
            self.stu_head = student.classifier

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
            z = torch.randn(self.synthesis_batch_size, self.nz).to(self.device)
            # print(z)
            G.train()
            self.optimizers[l].zero_grad()            
            mu_theta = G(z, l=l)
            samples = self.normalizer(mu_theta)
            # print(samples)
            x_inputs = self.normalizer(samples, reverse=True)
            
            t_out, t_feat = self.teacher(samples, l=l, return_features=True)
            p = F.softmax(t_out / self.T, dim=1).mean(0)
            ent = -(p*p.log()).sum()
            loss_oh = F.cross_entropy( t_out, t_out.max(1)[1])
            loss_act = - t_feat.abs().mean()
            if self.bn > 0:
                loss_bn = sum([h.r_feature for h in self.hooks])
            else:
                loss_bn = torch.zeros(1).to(self.device)
            
            # Negative Divergence.
            if self.adv > 0:
                s_out, s_feat = self.student(samples, l=l, return_features=True)
                if self.adv_type == 'js':
                    l_js = jsdiv(s_out, t_out, T=3)
                    loss_adv = 1.0-torch.clamp(l_js, 0.0, 1.0)
                if self.adv_type == 'kl':
                    mask = (s_out.max(1)[1]==t_out.max(1)[1]).float()
                    loss_adv = -(kldiv(s_out, t_out, reduction='none', T=3).sum(1) * mask).mean()
            else:
                loss_adv = torch.zeros(1).to(self.device)
            
            loss = self.lmda_ent * ent + self.adv * loss_adv+ self.oh * loss_oh + self.act * loss_act + self.bn * loss_bn
            if self.hard > 0:
                if self.adv == 0:
                    s_out, s_feat = self.student(samples, l=l, return_features=True)
                if t_feat.size()[-1] != s_feat.size()[-1]:
                    project_layer = torch.nn.Linear(s_feat.size()[-1], t_feat.size()[-1]).to(self.device)
                    # print(project_layer)
                    s_feat = project_layer(s_feat)
                loss_cnce = self.neg_bank(t_feat, s_feat, hard_factor, length=self.k)
                loss += self.hard * loss_cnce
            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = mu_theta
                    
            loss.backward()
            self.optimizers[l].step()

        # print(best_inputs)
        # if self.memory:
        #     self.update_loader(best_inputs=best_inputs)
        
        # self.student.train()
        # print(best_inputs)
        return {'synthetic': best_inputs}

    @torch.no_grad()
    def sample(self, l=0, warmup=True):
        if not self.memory:
            self.G_list[l].eval() 
            z = torch.randn( size=(self.sample_batch_size, self.nz), device=self.device )
            # print(z)
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,), device=self.device)
            inputs = self.G_list[l](z, l=l)
            # print(inputs)
        else:
            inputs = self.data_iter.next()
            
        return inputs

    def update_loader(self, best_inputs):
        if self.mode == 'memory':
            self.data_pool.add(best_inputs)
            if self.memory:
                dst = self.data_pool.get_dataset(transform=self.transform)
                # dst = self.data_pool.get_dataset()
                if self.distributed:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(dst)
                else:
                    train_sampler = None
                loader = torch.utils.data.DataLoader(
                    dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
                    num_workers=4, pin_memory=True, sampler=train_sampler)
                self.data_iter = DataIter(loader)