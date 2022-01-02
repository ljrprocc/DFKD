import torch
import torch.nn as nn
import yaml
import argparse
import os
import sys
import time
from utils.os_use import add_dict
from torch.backends import cudnn
from pytorchcv.model_provider import get_model as ptcv_get_model
from pytorchcv.generator import Generator, Generator_imagenet
from dataloader import DataLoader
from trainer import Trainer
import logging


class Experiment:
    def __init__(self, opt, G):
        self.opt = opt
        self.G = G
        self.optimizer_state=None
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.visible_devices
        self.save_path = './save/test1/'
        self.logger = self.set_logger()
        self.prepare()

    def prepare(self):
        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self.logger.info(self.model)
        self._set_trainer()

    def _set_gpu(self):
        torch.manual_seed(self.opt.seed)
        torch.cuda.maunal_seed(self.opt.seed)
        assert self.opt.GPU <= torch.cuda.device_count() - 1
        cudnn.benchmark = True

    def set_logger(self):
        logger = logging.getLogger('baseline')
        file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        console_formatter = logging.Formatter('%(message)s')
        # file log
        file_handler = logging.FileHandler(os.path.join(self.settings.save_path, "train_test.log"))
        file_handler.setFormatter(file_formatter)

        # console log
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.setLevel(logging.INFO)
        return logger

    def _set_trainer(self):
        self.trainer = Trainer(
            models=(self.model, self.model_t, self.G),
            loaders=(self.train_loader, self.test_loader),
            settings=self.opt,
            logger=self.logger,
            optimizer_state=self.optimizer_state
        )


    def _set_dataloader(self):
        dataloader = DataLoader(dataset=self.opt.dataset, batch_size=self.opt.batch_size, data_path=self.opt.root, n_threads=self.opt.num_workers, ten_crop=self.opt.ten_crop, logger=self.logger)

        self.train_loader, self.test_loader = dataloader.getloader()

    def _set_model(self):
        if dataset in ['cifar100', 'cifar10']:
            if args.network_s in ['resnet20_cifar100', 'resnet20_cifar10']:
                self.model = ptcv_get_model(args.network_s, pretrained=True)
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


    def train(self):
        best_top1 = 100
        best_top5 = 100
        st_time = time.time()

    def eval(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DFKD Baseline')
    parser.add_argument('--config_path', type=str, default='configs/cifar10.yaml', help='The path to the config file about model')
    parser.add_argument('--gpu', type=str, default='0', help='The gpu(s) used for training')
    parser.add_argument('--eval', action="store_true", help='Flag for evaluation.')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--random_seed', type=int, default=1234, help='Manual random seed for random operation.')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        opt = yaml.load(f)

    args = add_dict(args, opt)
    
    
    dataset = args.config_path.split('/')[:-5]
    args.dataset = dataset
    exp = Experiment(opt=args)
    # Trainer: models, loaders, option, tb_logger, ckpt_load
    # 1. models
    if dataset in ['cifar100', 'cifar10']:
        if args.network in ['resnet20_cifar100', 'resnet20_cifar10']:
            weight_t = ptcv_get_model(args.network, pretrained=True).output.weight.detach()
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


    if args.eval:
        exp.eval()
    else:
        exp.train()

    