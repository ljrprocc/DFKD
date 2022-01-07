import torch
import torch.nn as nn
import datetime
import yaml
import argparse
import os
import sys
import time
from utils.os_use import add_dict
from torch.backends import cudnn
from pytorchcv.model_provider import get_model as ptcv_get_model
from pytorchcv.generator import Generator, Generator_imagenet
from dataloader import DataLoader, get_train_loader
from trainer import Trainer
from tensorboardX import SummaryWriter
import logging

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Experiment:
    def __init__(self, opt, G):
        self.opt = opt
        self.G = G
        self.optimizer_state=None
        self.start_epoch = opt.start_epoch
        self.n_epochs = opt.n_epochs
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpu
        self.save_path = './save/{}/'.format(opt.save_name)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.logger = self.set_logger()
        self.prepare()

    def prepare(self):
        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self.logger.info(self.model)
        self._set_trainer()

    def _set_gpu(self):
        torch.manual_seed(self.opt.random_seed)
        torch.cuda.manual_seed(self.opt.random_seed)
        gpu_lists = self.opt.gpu.split(',')
        assert len(gpu_lists) <= torch.cuda.device_count()
        cudnn.benchmark = True

    def set_logger(self):
        logger = logging.getLogger('baseline')
        file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        console_formatter = logging.Formatter('%(message)s')
        # file log
        file_handler = logging.FileHandler(os.path.join(self.save_path, "train_test.log"))
        file_handler.setFormatter(file_formatter)

        # console log
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.setLevel(logging.INFO)
        if self.opt.set_tb_logger:
            self.tb_logger = SummaryWriter(logdir='./logs/{}/'.format(self.opt.save_name))
        return logger

    def _set_trainer(self):
        self.trainer = Trainer(
            models=(self.model, self.model_t, self.G),
            loaders=(self.train_loader, self.test_loader),
            settings=self.opt,
            logger=self.logger,
            tb_logger=self.tb_logger,
            optimizer_state=self.optimizer_state
        )


    def _set_dataloader(self):
        dataloader = DataLoader(dataset=self.opt.dataset, batch_size=self.opt.batch_size, data_path=self.opt.root, n_threads=self.opt.num_workers, ten_crop=self.opt.ten_crop, logger=self.logger)

        self.train_loader, self.test_loader = dataloader.getloader()

    def _set_model(self):
        if dataset in ['cifar100', 'cifar10']:
            if args.network_s in ['resnet20_cifar100', 'resnet20_cifar10']:
                # self.model = ptcv_get_model(args.network_s, pretrained=True)
                self.model = ptcv_get_model(args.network_s)
                # self.model.apply(weights_init)
                # print('********************')
                self.model_t = ptcv_get_model(args.network, pretrained=True)
                self.model_t.eval()

            else:
                assert False, 'unsupport network: ' + args.network
        elif dataset in ['imagenet']:
            if args.network_s in ['resnet18', 'resnet50']:
                self.model = ptcv_get_model(args.network_s, pretrained=True)
                self.model_t = ptcv_get_model(args.network, pretrained=True)
                self.model_t.eval()
                
            else:
                assert False, 'unsupport network: ' + args.network
        else:
            assert False, "invalid data set"

    def freeze_model(self,model):
        """
        freeze the activation range
        """
        if type(model) == nn.Sequential:
            for n, m in model.named_children():
                self.freeze_model(m)
        else:
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    self.freeze_model(mod)
            return model

    def unfreeze_model(self,model):
        """
        unfreeze the activation range
        """
        if type(model) == nn.Sequential:
            for n, m in model.named_children():
                self.unfreeze_model(m)
        else:
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    self.unfreeze_model(mod)
            return model


    def train(self):
        
        st_time = time.time()
        best_ep = 0

        test_error, test_loss, test5_error = self.trainer.test_teacher()
        for epoch in range(self.start_epoch, self.n_epochs):
            self.epoch = epoch
            # if epoch < 4:
            #     print('Unfreeze model')
            #     self.unfreeze_model(self.model)
            train_error, train_loss, train5_error = self.trainer.train_loop(epoch=epoch)
            # self.freeze_model(self.model)
            if self.opt.dataset in ["cifar100","cifar10"]:
                test_error, test_loss, test5_error = self.trainer.test_stu(log=True, epoch=epoch)
            elif self.opt.dataset in ["imagenet"]:
                if epoch > self.opt.warmup_epochs - 2:
                    test_error, test_loss, test5_error = self.trainer.test_stu(log=True, epoch=epoch)
                else:
                    test_error = 100
                    test5_error = 100
            else:
                assert False, "invalid data set"


            if best_top1 >= test_error:
                best_ep = epoch+1
                best_top1 = test_error
                best_top5 = test5_error
                print('Saving a best checkpoint ...')
                torch.save(self.trainer.model.state_dict(),f"{self.save_path}/student_model_{self.opt.dataset}-{self.opt.network}-w{self.opt.network_s}.pt")
                torch.save(self.trainer.G.state_dict(),f"{self.save_path}/generator_{self.opt.dataset}-{self.opt.network}-w{self.opt.network_s}.pt")

        self.logger.info("#==>Best Result of ep {:d} is: Top1 Error: {:f}, Top5 Error: {:f}, at ep {:d}".format(epoch+1, best_top1, best_top5, best_ep))
        self.logger.info("#==>Best Result of ep {:d} is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f} at ep {:d}".format(epoch+1 , 100 - best_top1, 100 - best_top5, best_ep))

        end_time = time.time()
        time_interval = end_time - st_time
        t_string = "Running Time is: " + str(datetime.timedelta(seconds=time_interval)) + "\n"
        self.logger.info(t_string)

        return best_top1, best_top5

    def train_two_stage(self):
        # train_G
        # best_top1 = 100
        # best_top5 = 100
        # st_time = time.time()
        # best_ep = 0
        self.train_G(self)
        self.trainer.generate_batch(save_path=self.save_path, n=self.opt.s_iter, train_s=True)

    def train_G(self):
        for epoch in range(self.start_epoch, self.n_epochs):
            self.trainer.train_G(epoch)
            if epoch % self.opt.save_freq==0:
                torch.save(self.trainer.G.state_dict(),f"{self.save_path}/generator_{self.opt.dataset}-{self.opt.network}-w{self.opt.network_s}_epoch{epoch}.pt")
        torch.save(self.trainer.G.state_dict(),f"{self.save_path}/generator_{self.opt.dataset}-{self.opt.network}-w{self.opt.network_s}.pt")

    def generate_batch(self):
        self.trainer.generate_batch(save_path=self.save_path, n=self.opt.s_iter, train_s=False)

    def train_S(self):
        best_ep = 0
        best_top1 = 100
        best_top5 = 100
        st = time.time()
        train_loader = get_train_loader(self.opt)
        for i in range(self.opt.s_epochs):
            self.trainer.train_stu(i, train_loader)
            if self.opt.dataset in ["cifar100","cifar10"]:
                test_error, test_loss, test5_error = self.trainer.test_stu(log=True, epoch=i)
            elif self.opt.dataset in ["imagenet"]:
                if i > self.opt.warmup_epochs - 2:
                    test_error, test_loss, test5_error = self.trainer.test_stu(log=True, epoch=i)
                else:
                    test_error = 100
                    test5_error = 100
            else:
                assert False, "invalid data set"
            if best_top1 >= test_error:
                best_ep = i+1
                best_top1 = test_error
                best_top5 = test5_error
                print('Saving a best checkpoint ...')
                torch.save(self.trainer.model.state_dict(),f"{self.save_path}/student_model_{self.opt.dataset}-{self.opt.network}-w{self.opt.network_s}_best.pt")
            if i % self.save_freq == 0:
                torch.save(self.trainer.model.state_dict(),f"{self.save_path}/student_model_{self.opt.dataset}-{self.opt.network}-w{self.opt.network_s}-{i}.pt")

            self.logger.info("#==>Best Result of ep {:d} is: Top1 Error: {:f}, Top5 Error: {:f}, at ep {:d}".format(i+1, best_top1, best_top5, best_ep))
            self.logger.info("#==>Best Result of ep {:d} is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f} at ep {:d}".format(i+1 , 100 - best_top1, 100 - best_top5, best_ep))

        torch.save(self.trainer.model.state_dict(),f"{self.save_path}/student_model_{self.opt.dataset}-{self.opt.network}-w{self.opt.network_s}_last.pt")
        end_time = time.time()
        time_interval = end_time - st
        t_string = "Running Time is: " + str(datetime.timedelta(seconds=time_interval)) + "\n"
        self.logger.info(t_string)


    def eval(self):
        weight_path = f"{self.opt.ckpt_path}/student_model_{self.opt.dataset}-{self.opt.network}-w{self.opt.network_s}.pt"
        self.trainer.model.load_state_dict(torch.load(weight_path))
        self.trainer.test_student()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DFKD Baseline')
    parser.add_argument('--config_path', type=str, default='configs/cifar10.yaml', help='The path to the config file about model')
    parser.add_argument('--gpu', type=str, default='0', help='The gpu(s) used for training')
    parser.add_argument('--eval', action="store_true", help='Flag for evaluation.')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--random_seed', type=int, default=1234, help='Manual random seed for random operation.')
    parser.add_argument('--stage', type=str, default='joint', choices=['joint', 'two_stage', 'train_g', 'train_s', 'generate_batch'])
    parser.add_argument('--df_folder', type=str, default=None, help='2nd stage root dir')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        opt = yaml.load(f)

    args = add_dict(args, opt)
    
    
    dataset = args.config_path.split('/')[-1][:-5]
    args.dataset = dataset
    
    # Trainer: models, loaders, option, tb_logger, ckpt_load
    # 1. models
    # print(dataset)
    if dataset in ['cifar100', 'cifar10']:
        if args.network in ['resnet20_cifar100', 'resnet20_cifar10', 'resnet56_cifar100', 'resnet56_cifar10']:
            weight_t = ptcv_get_model(args.network, pretrained=True).output.weight.detach()
            if args.random_embedding:
                weight_t = None
            generator = Generator(args, teacher_weight=weight_t, freeze=args.freeze)

        else:
            assert False, 'unsupport network: ' + args.network
    elif dataset in ['imagenet']:
        if args.network in ['resnet18', 'resnet50']:
            weight_t = ptcv_get_model(args.network, pretrained=True).output.weight.detach()
            generator = Generator_imagenet(args, teacher_weight=weight_t, freeze=args.freeze)
        else:
            assert False, 'unsupport network: ' + args.network
    else:
        assert False, "invalid data set"

    exp = Experiment(opt=args, G=generator)
    if args.eval:
        exp.eval()
    else:
        choice = args.stage
        if choice == 'two_stage':
            exp.train_two_stage()
        elif choice == 'joint':
            exp.train()
        elif choice == 'train_g':
            exp.train_G()
        elif choice == 'generate_batch':
            exp.generate_batch()
        elif choice == 'train_s':
            exp.train_S()

    