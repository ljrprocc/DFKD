import torch
import datafree
import copy

class Environment():
    def __init__(self, models, evaluator=None, batch_size=128, latent_dim=100, device='cpu'):
        self.teacher, self.student, self.G = models
        self.dyn_student = copy.deepcopy(self.student)
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.device = device
        self.teacher.eval()
        self.dyn_student.train()
        # self.student.eval()
        # self.state = self.reset()
        self.evaluator = evaluator
        

    def step(self, action, supervised=True):
        # Compute the target reward for the given state.
        state = self.state
        self.dyn_student.train()
        criterion = datafree.criterions.KLDiv(T=1, reduction='none')
        if supervised:
            # Reward = Topk(s_{k'}) - Topk(s_{k})
            # Here Topk() includes the iteration of validation set, maybe not strictly data-free.
            self.optimizer.zero_grad()
            # print(action)
            # print('**********')
            # print(self.s_out)
            loss = criterion(self.s_out, self.t_out).sum(1)
            # print(loss)
            real_loss = (action * loss).mean()
            real_loss.backward()
            self.optimizer.step()
            self.dyn_student.eval()
            eval_result_previous = self.evaluator(self.student, device=self.device)
            eval_result_now = self.evaluator(self.dyn_student, device=self.device)
            top1_previous, top5_previous = eval_result_previous['Acc']
            top1_now, top5_now = eval_result_now['Acc']
            reward = (top1_now - top1_previous) + 0.1 * (top5_now - top5_previous)
            self.s_out, new_s_feat = self.dyn_student(self.input.detach(), return_features=True)
            # self.s_out = new_s_out
            self.state = torch.cat([self.t_feat, new_s_feat], 1)
            return reward, self.state.detach(), reward > 0
        else:
            # Reward = diversity(s_{k'}) - diversity(s_{k})
            # Expected performance improvement measured by some unsupervised settings.
            pass
                


    def reset(self, input=None):
        
        self.dyn_student.train()
        self.optimizer = torch.optim.SGD(self.dyn_student.parameters(), lr=0.1, weight_decay=0.0, momentum=0.9)
        # for k, v in self.evaluator.items():
        #     v.reset()
        if input is None:
            z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
            self.input = self.G(z)
        else:
            self.input = input
        with torch.no_grad():
            self.t_out, self.t_feat = self.teacher(self.input.detach(), return_features=True)
        self.s_out, s_feat = self.dyn_student(self.input.detach(), return_features=True)
        # print(self.dyn_student(self.input))
        # print(self.s_out)
        # print(t_feat.shape, s_feat.shape)
        self.state = torch.cat([self.t_feat, s_feat], 1)

