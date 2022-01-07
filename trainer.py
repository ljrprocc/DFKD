import time
import torch
from torch import nn
from torch.optim import optimizer
from torch.optim.lr_scheduler import ExponentialLR
from pytorchcv.models.diaresnet import DIALSTMCell
from utils.model_transform import data_parallel
from utils.compute import AverageMeter, compute_singlecrop, compute_tencrop
from torch.autograd import Variable, backward
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import tqdm
import os

__all__ = ['Trainer']


class Trainer(nn.Module):
    def __init__(self, models, loaders, settings, logger, tb_logger=None, optimizer_state=None):
        super(Trainer, self).__init__()
        model, model_t, model_G = models
        self.settings = settings
        self.model = data_parallel(model, self.settings.nGPU, self.settings.gpu.split(','))
        self.model_t = data_parallel(model_t, self.settings.nGPU, self.settings.gpu.split(','))

        self.all_y = torch.arange(self.settings.n_cls).cuda()

        self.no_noise = torch.zeros(self.settings.n_cls, self.settings.latent_dim).cuda()

        self.G = data_parallel(model_G, self.settings.nGPU, self.settings.gpu.split(','))

        self.train_loader, self.test_loader = loaders
        self.tb_logger = tb_logger

        self.log_soft = nn.LogSoftmax(dim=1)
        self.mse_loss = nn.MSELoss().cuda()

        self.logger = logger
        self.scalar_info = {}
        self.mean_list = []
        self.var_list = []
        self.t_running_mean = []
        self.t_running_var = []
        self.fix_G = False

        # self.optimizer_S = torch.optim.Adam(
        #     params=self.model.parameters(),
        #     lr=self.settings.lr_s,
        #     betas=(0.9, 0.999),
        #     eps=1e-5,
        #     weight_decay=self.settings.weight_decay
        # )
        self.optimizer_S = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.settings.lr_s,
            momentum=0.9,
            weight_decay=self.settings.weight_decay,
            nesterov=True
        )

        if optimizer_state is not None:
            self.optimizer_S.load_state_dict(optimizer_state)

        self.optimizer_G = torch.optim.Adam(
            params=self.G.parameters(),
            lr=self.settings.lr_g,
            betas=(self.settings.beta_g1, self.settings.beta_g2),
            weight_decay=self.settings.weight_decay
        )

        self.scheduler_S = ExponentialLR(optimizer=self.optimizer_S, gamma=0.9)
        self.scheduler_G = ExponentialLR(optimizer=self.optimizer_G, gamma=0.9)
    
    def loss_fn_kd(self, logits, y, teacher_logits, linear=None):
        # criterion_d = nn.CrossEntropyLoss(reduction='none').cuda()
        kdloss = nn.KLDivLoss().cuda()

        alpha = self.settings.alpha
        t = self.settings.temperature
        a = F.log_softmax(logits / t, dim=1)
        b = F.softmax(teacher_logits / t, dim=1)
        c = alpha * t * t

        # d = criterion_d(logits, y).mean()
        d = (-(linear*self.log_soft(logits)).sum(1)).mean()

        l_kd = kdloss(a, b) * c + d
        # print(kdloss(a, b) * c, d)
        return l_kd

    def hook_fn_forward(self, module, input, output):
        input = input[0]
        mean = input.mean(dim=[0,2,3])
        var = torch.var(input, dim=[0,2,3], unbiased=False)

        self.mean_list.append(mean)
        self.var_list.append(var)

        self.t_running_mean.append(module.running_mean)
        self.t_running_var.append(module.running_var)

    def forward(self, x, t_logits, labels=None, linear=None):
        output = self.model(x)
        if labels is not None:
            loss = self.loss_fn_kd(output, labels, t_logits, linear)
            return output, loss
        else:
            return output, None

    def backward_G(self, loss_G):
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

    def backward_S(self, loss_S):
        self.optimizer_S.zero_grad()
        loss_S.backward()
        self.optimizer_S.step()
    
    def train_loop(self, epoch):
        # self.scalar_info.clear()
        top1_error = AverageMeter()
        top1_loss = AverageMeter()
        top5_error = AverageMeter()
        fp_acc = AverageMeter()

        # Set training flag for the student model
        

        start_time = time.time()
        st = start_time

        iters = self.settings.total_iters

        if epoch == 0:
            for m in self.model_t.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.register_forward_hook(self.hook_fn_forward)

        self.scheduler_S.step()
        self.scheduler_G.step()
        for i in range(iters):
            
            self.model.eval()
            self.model_t.eval()
            self.G.train()
            start_time = time.time()
            # data_time = start_time - end_time
            z = Variable(torch.randn(self.settings.batch_size, self.settings.latent_dim)).cuda()
            labels = Variable(torch.randint(0, self.settings.n_cls, (self.settings.batch_size, ))).cuda()
            z = z.contiguous()
            labels = labels.contiguous()
            images = self.G(z, labels)
            label_loss = Variable(torch.zeros(self.settings.batch_size, self.settings.n_cls)).cuda()
            label_loss.scatter_(1, labels.unsqueeze(1), 1.0)

            self.mean_list.clear()
            self.var_list.clear()

            logit_teacher = self.model_t(images)
            loss_one_hot = (-(label_loss*self.log_soft(logit_teacher)).sum(dim=1)).mean()
            
            BNS_loss = torch.zeros(1).cuda()

            for num in range(len(self.mean_list)):
                BNS_loss += self.mse_loss(self.mean_list[num], self.t_running_mean[num]) + self.mse_loss(self.var_list[num], self.t_running_var[num])

            BNS_loss = BNS_loss / len(self.mean_list)

            loss_G = loss_one_hot + 0.1 * BNS_loss
            self.backward_G(loss_G)
            
            output, loss_S = self.forward(images.detach(), logit_teacher.detach(), labels=labels, linear=label_loss)
            

            if epoch > self.settings.warmup_epochs:
                self.model.train()
                self.model_t.eval()
                self.G.eval()
                self.backward_S(loss_S)

            single_error, single_loss, single5_error = compute_singlecrop(outputs=output, labels=labels, loss=loss_S, top5_flag=True, mean_flag=True)

            top1_error.update(single_error, images.size(0))
            top1_loss.update(single_loss, images.size(0))
            top5_error.update(single5_error, images.size(0))

            end_time = time.time()

            gt = labels.data.cpu().numpy()
            d_acc = np.mean(np.argmax(logit_teacher.data.cpu().numpy(), 1) == gt)

            fp_acc.update(d_acc)

            if i % self.settings.print_freq == 0:
                print("[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%] [G loss: %.6f] [Oe-hot loss: %.6f] [BNS_loss:%.6f] [S loss: %.6f] [Time: %.6f s]" % (epoch + 1, self.settings.n_epochs, i+1, iters, 100 * fp_acc.avg, loss_G.item(), loss_one_hot.item(), BNS_loss.item(), loss_S.item(), time.time() - st))
                # print(np.argmax(logit_teacher.data.cpu().numpy(), 1), gt)
                # print(top1_error.avg)

                global_iter = epoch * iters + i

                self.scalar_info['accuracy every epoch'] = 100 * d_acc
                self.scalar_info['G loss every epoch'] = loss_G
                self.scalar_info['One-hot loss every epoch'] = loss_one_hot
                self.scalar_info['S loss every epoch'] = loss_S

                self.scalar_info['training_top1error'] = top1_error.avg
                self.scalar_info['training_top5error'] = top5_error.avg
                self.scalar_info['training_loss'] = top1_loss.avg
                
                if self.tb_logger is not None:
                    for tag, value in list(self.scalar_info.items()):
                        self.tb_logger.add_scalar(tag, value, global_iter)
                    self.scalar_info = {}

        

        return top1_error.avg, top1_loss.avg, top5_error.avg


    def test_stu(self, log=False, epoch=0):
        top1_error = AverageMeter()
        top1_loss = AverageMeter()
        top5_error = AverageMeter()
        
        self.model.eval()
        self.model_t.eval()
        
        iters = len(self.test_loader)
        start_time = time.time()
        st = start_time
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                start_time = time.time()
                
                labels = labels.cuda()
                images = images.cuda()
                output = self.model(images)
                
                loss = torch.ones(1)
                self.mean_list.clear()
                self.var_list.clear()
                
                single_error, single_loss, single5_error = compute_singlecrop(
					outputs=output, loss=loss,
					labels=labels, top5_flag=True, mean_flag=True)

                if False:
                    print( "[Batch %d/%d] [acc: %.4f%%]" % (i + 1, iters, (100.00-top1_error.avg)))

                top1_error.update(single_error, images.size(0))
                top1_loss.update(single_loss, images.size(0))
                top5_error.update(single5_error, images.size(0))
				
                end_time = time.time()

        print( "Testing finished in %f s."%(time.time() - st))
        print( "Student Model Accuracy : %.4f%%" % (100.00-top1_error.avg))
        if log:
            self.tb_logger.add_scalar('test_top1_error', top1_error.avg, epoch)
            self.tb_logger.add_scalar('test_top1_loss', top1_loss.avg, epoch)
            self.tb_logger.add_scalar('test_top5_error', top5_error.avg, epoch)
        return top1_error.avg, top1_loss.avg, top5_error.avg

    def train_G(self, epoch):
        
        fp_acc = AverageMeter()

        # Set training flag for the student model
        start_time = time.time()
        st = start_time

        iters = self.settings.total_iters

        if epoch == 0:
            for m in self.model_t.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.register_forward_hook(self.hook_fn_forward)

        self.scheduler_S.step()
        self.scheduler_G.step()
        for i in range(iters):
            
            # self.model.eval()
            self.model_t.eval()
            self.G.train()
            start_time = time.time()
            # data_time = start_time - end_time
            z = Variable(torch.randn(self.settings.batch_size, self.settings.latent_dim)).cuda()
            labels = Variable(torch.randint(0, self.settings.n_cls, (self.settings.batch_size, ))).cuda()
            z = z.contiguous()
            labels = labels.contiguous()
            images = self.G(z, labels)
            label_loss = Variable(torch.zeros(self.settings.batch_size, self.settings.n_cls)).cuda()
            label_loss.scatter_(1, labels.unsqueeze(1), 1.0)

            self.mean_list.clear()
            self.var_list.clear()

            logit_teacher = self.model_t(images)
            loss_one_hot = (-(label_loss*self.log_soft(logit_teacher)).sum(dim=1)).mean()
            
            BNS_loss = torch.zeros(1).cuda()

            for num in range(len(self.mean_list)):
                BNS_loss += self.mse_loss(self.mean_list[num], self.t_running_mean[num]) + self.mse_loss(self.var_list[num], self.t_running_var[num])

            BNS_loss = BNS_loss / len(self.mean_list)

            loss_G = loss_one_hot + 0.1 * BNS_loss
            gt = labels.data.cpu().numpy()
            # print(logit_teacher.argmax(1))
            d_acc = np.mean(np.argmax(logit_teacher.data.cpu().numpy(), 1) == gt)
            fp_acc.update(d_acc)

            self.backward_G(loss_G)
            if i % self.settings.print_freq == 0:
                print("[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%] [G loss: %.6f] [Oe-hot loss: %.6f] [BNS_loss:%.6f] [Time: %.6f s]" % (epoch + 1, self.settings.n_epochs, i+1, iters, 100 * fp_acc.avg, loss_G.item(), loss_one_hot.item(), BNS_loss.item(), time.time() - st))
                # print(np.argmax(logit_teacher.data.cpu().numpy(), 1), gt)
                # print(top1_error.avg)

                global_iter = epoch * iters + i

                self.scalar_info['accuracy every epoch'] = 100 * d_acc
                self.scalar_info['G loss every epoch'] = loss_G
                self.scalar_info['One-hot loss every epoch'] = loss_one_hot
                if self.tb_logger is not None:
                    for tag, value in list(self.scalar_info.items()):
                        self.tb_logger.add_scalar(tag, value, global_iter)
                    self.scalar_info = {}

    def train_stu_loop(self, images, labels, i, n, st):
        top1_error = AverageMeter()
        top1_loss = AverageMeter()
        top5_error = AverageMeter()
        fp_acc = AverageMeter()
        logit_teacher = self.model_t(images)
        label_loss = Variable(torch.zeros(self.settings.batch_size, self.settings.n_cls)).cuda()
        label_loss.scatter_(1, labels.unsqueeze(1), 1.0)
        
        output, loss_S = self.forward(images.detach(), logit_teacher, labels=labels, linear=label_loss)
        print(loss_S)
        self.backward_S(loss_S)
        gt = labels.data.cpu().numpy()
        d_acc = np.mean(np.argmax(logit_teacher.data.cpu().numpy(), 1) == gt)

        single_error, single_loss, single5_error = compute_singlecrop(outputs=output, labels=labels, loss=loss_S, top5_flag=True, mean_flag=True)

        top1_error.update(single_error, images.size(0))
        top1_loss.update(single_loss, images.size(0))
        top5_error.update(single5_error, images.size(0))

        fp_acc.update(d_acc)

        if i % self.settings.print_freq:
            top1_err, top1_ls, top5_err = self.test_stu(log=True, epoch=i)
            print("[Iter %d/%d] [S loss: %.6f] [Train Stu Acc: %.4f%%] [Test top1 acc: %.4f%%] [Time: %.6f s] " % ( i+1, n, loss_S.item(), fp_acc.avg, 100 - top1_err, time.time() - st))

        return top1_error.avg, top1_loss.avg, top5_error.avg

    def generate_batch(self, n=100000, save_path=None, save=True, train_s=False):
        self.model.train()
        self.model_t.eval()
        self.G.eval()
        st = time.time()

        for i in tqdm.tqdm(range(n)):         
            with torch.no_grad():
                z = torch.randn(1, self.settings.latent_dim).cuda()
                labels = torch.randint(0, self.settings.n_cls, (1,)).cuda()
                images = self.G(z, labels)
                if save:
                    if not os.path.exists('{}/save/'.format(save_path)):
                        os.mkdir('{}/save/'.format(save_path))
                    vutils.save_image(images, '{}/save/label_{}_id_{}.jpg'.format(save_path, labels.item(), i), normalize=True, nrow=1, padding=0, range=2)
            if not train_s:
                continue
            top1_err, top1_ls, top5_err = self.train_stu_loop(images, labels, i, n, st)     
        return top1_err, top1_ls, top5_err

    def train_stu(self, epoch, train_loader):
        st = time.time()
        print('**********epoch {} / {}***********'.format(epoch, self.settings.n_epochs))
        for i, (x, y) in enumerate(train_loader):
            images = x.cuda()
            labels = y.cuda()
            n = len(train_loader)
            top1_err, top1_ls, top5_err = self.train_stu_loop(images, labels, i, n, st) 

        return top1_err, top1_ls, top5_err
	
    def test_teacher(self):
        top1_error = AverageMeter()
        top1_loss = AverageMeter()
        top5_error = AverageMeter()
        
        self.model_t.eval()
        start_time = time.time()
        st = start_time
        iters = len(self.test_loader)

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                start_time = time.time()
                # data_time = start_time - end_time

                labels = labels.cuda()
                if self.settings.ten_crop:
                    image_size = images.size()
                    images = images.view(
                        image_size[0] * 10, image_size[1] / 10, image_size[2], image_size[3])
                    images_tuple = images.split(image_size[0])
                    output = None
                    for img in images_tuple:
                        if self.settings.nGPU == 1:
                            img = img.cuda()
                        img_var = Variable(img, volatile=True)
                        temp_output, _ = self.forward(img_var)
                        if output is None:
                            output = temp_output.data
                        else:
                            output = torch.cat((output, temp_output.data))
                    single_error, single_loss, single5_error = compute_tencrop(
                        outputs=output, labels=labels)
                else:
                    if self.settings.nGPU == 1:
                        images = images.cuda()

                    output = self.model_t(images)

                    loss = torch.ones(1)
                    self.mean_list.clear()
                    self.var_list.clear()

                    single_error, single_loss, single5_error = compute_singlecrop(
                        outputs=output, loss=loss,
                        labels=labels, top5_flag=True, mean_flag=True)
                #
                top1_error.update(single_error, images.size(0))
                top1_loss.update(single_loss, images.size(0))
                top5_error.update(single5_error, images.size(0))

                end_time = time.time()
                iter_time = end_time - start_time

        print( "Testing finished in %f s."%(time.time() - st))
        print(
                "Teacher network: [Batch %d/%d] [acc: %.4f%%]"
                % (i + 1, iters, (100.00 - top1_error.avg))
        )

        # self.run_count += 1

        return top1_error.avg, top1_loss.avg, top5_error.avg
