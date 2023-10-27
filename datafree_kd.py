import argparse
from math import gamma
from multiprocessing import reduction
import os
import random
import shutil
import time
from datetime import timedelta
import warnings

import registry
import datafree
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from datafree.utils import difficulty_mining


parser = argparse.ArgumentParser(description='Data-free Knowledge Distillation')

# Data Free
parser.add_argument('--method', required=True, choices=['zskt', 'dfad', 'dafl', 'deepinv', 'dfq', 'cmi', 'zskd', 'dfme', 'softtarget', 'cudfkd', 'pretrained', 'adadfkd', "fast_meta"])
parser.add_argument('--adv', default=0, type=float, help='scaling factor for adversarial distillation')
parser.add_argument('--adv_type',choices=['js', 'kl'], default='js', help='Adversirial training for which divergence.')
parser.add_argument('--cond', action="store_true", help='using class-conditional generation strategy.')
parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
parser.add_argument('--oh', default=0, type=float, help='scaling factor for one hot loss (cross entropy)')
parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
parser.add_argument('--l1', default=0.01, type=float, help='scaling factor for l1-alignment at teacher space.')
parser.add_argument('--balance', default=0, type=float, help='scaling factor for class balance')
parser.add_argument('--depth', default=2, type=int, help='Depth of DCGAN-type Generator.')
parser.add_argument('--no_feature', action="store_true", help="Flag for whether use feature map distribution alignment.")
parser.add_argument('--only_feature', action="store_true", help="Flag for whether use only last feature map distribution alignment.")
parser.add_argument('--save_dir', default='run/synthesis', type=str)
parser.add_argument('--no_logits', type=int, default=1)
parser.add_argument('--logit_correction', type=str, default='mean', choices=['none', 'mean'])
parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'kl', 'l2'])
parser.add_argument('--grad_m', type=int, default=1, help='Number of steps to approximate the gradients')
parser.add_argument('--grad_epsilon', type=float, default=1e-3) 
parser.add_argument('--lmda_ent', default=0.1, type=float, help='Scaling factor for entropy minimization.')
parser.add_argument('--L', default=2, type=int, help='The depth for generation.')
parser.add_argument('--grad_adv', default=0.2, type=float, help='the gradient for adding lambda_adv.')
parser.add_argument('--begin_fraction', default=0.25, type=float, help='Begin epoch for open the adversarial training.')
parser.add_argument('--end_fraction', default=0.75, type=float, help='End epoch for open the adversarial training.')

parser.add_argument('--forward_differences', type=int, default=1, help='Always set to 1')


parser.add_argument('--cr', default=1, type=float, help='scaling factor for contrastive model inversion')
parser.add_argument('--cr_T', default=0.5, type=float, help='temperature for contrastive model inversion')
parser.add_argument('--cmi_init', default=None, type=str, help='path to pre-inverted data')

# Basic
parser.add_argument('--data_root', default='data')
parser.add_argument('--teacher', default='wrn40_2')
parser.add_argument('--student', default='wrn16_1')
parser.add_argument('--dataset', default='cifar100')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate for KD')
parser.add_argument('--lr_decay_milestones', default="120,150,180", type=str,
                    help='milestones for learning rate decay')

parser.add_argument('--lr_g', default=1e-3, type=float, 
                    help='initial learning rate for generation')
parser.add_argument('--T', default=1, type=float)

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--g_steps', default=1, type=int, metavar='N',
                    help='number of iterations for generation')
# parser.add_argument('--g_steps_interval', default='2,1,1', type=str, help='number of iterations for generaton at each level of feature map distribution.')
parser.add_argument('--kd_steps', default=400, type=int, metavar='N',
                    help='number of iterations for KD after generation')
# parser.add_argument('--kd_steps_interval', type=str, default='400,200,100', help='number of iterations for KD after generaton for each level of feature map.')
parser.add_argument('--ep_steps', default=400, type=int, metavar='N',
                    help='number of total iterations in each epoch')
parser.add_argument('--save_freq', default=0, type=int, help='Save every t epochs for further visuali')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate_only', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--synthesis_batch_size', default=None, type=int,
                    metavar='N',
                    help='mini-batch size (default: None) for synthesis, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--reset_l0', default=0, type=int,
                    help='reset l0 in the generator during training')
parser.add_argument('--reset_bn', default=0, type=int,
                    help='reset bn layers during training')
parser.add_argument('--bn_mmt', default=0, type=float,
                    help='momentum when fitting batchnorm statistics')
parser.add_argument('--is_maml', default=1, type=int,
                    help='meta gradient: is maml or reptile')

parser.add_argument('--lr_z', default=1e-3, type=float, help='initial learning rate for latent code')
parser.add_argument('--warmup', default=0, type=int, metavar='N',
                    help='which epoch to start kd')


parser.add_argument('--log_y_kl', action="store_true", help='Flag for logging kl divergence at y space.')
parser.add_argument('--log_fidelity', action="store_true")
parser.add_argument('--memory', action="store_true")

# Difficulty sampler hyperparameters (for AdaDFKD)
parser.add_argument('--tau', default=10, type=float, help="temperature item for unsupervised curriculum sampling.")
parser.add_argument('--hard', default=1.0, type=float, help="hyperparmeter for hard curriculum sampling.")
parser.add_argument('--s_nce', default=0.0, type=float)
parser.add_argument('--mode', default='memory', type=str)
parser.add_argument('--mu', default=0.5, type=float)
parser.add_argument('--length', default=1.0, type=float)
parser.add_argument('--neg', default=0.1, type=float)
parser.add_argument('--N_neg', default=6144, type=int)
parser.add_argument('--debug', action="store_true", help="Visualization of anchor, positive and negative samples")

# Device
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
# TODO: Distributed and FP-16 training 
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--fp16', action='store_true',
                    help='use fp16')

# Misc
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--log_tag', default='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# Currcurilum Learning options
parser.add_argument('--curr_option', type=str, default='none')
parser.add_argument('--lambda_0', type=float, default=1.)
parser.add_argument('--nt2_mode', choices=['01', '02', '005', 'none', 'noisy'], default='none', type=str)
best_acc1 = 0
best_agg1 = 0
best_prob1 = 0

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.ngpus_per_node = ngpus_per_node = torch.cuda.device_count()
    print(ngpus_per_node)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        # args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.local_rank if args.distributed else args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, best_agg1, best_prob1
    args.gpu = gpu
    if args.distributed:
        args.local_rank = gpu
    else:
        args.local_rank = -1
    ############################################
    # GPU and FP16
    ############################################
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.local_rank == -1:
            args.local_rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.local_rank = args.rank * ngpus_per_node + gpu
        os.environ['MASTER_ADDR'] = "127.0.0.1"
        os.environ['MASTER_PORT'] = "6669"
        os.environ["RANK"] = str(args.local_rank)
        # os.environ["WORLD_SIZE"] = str(args.world_size)
        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.local_rank, timeout=timedelta(minutes=1))
    if args.fp16:
        from torch.cuda.amp import autocast, GradScaler
        args.scaler = GradScaler() if args.fp16 else None 
        args.autocast = autocast
    else:
        args.autocast = datafree.utils.dummy_ctx


    ############################################
    # Logger
    ############################################
    if args.log_tag != '':
        args.log_tag = '-'+args.log_tag
    if args.distributed:
        log_name = 'R%d-%s-%s-%s'%(args.local_rank, args.dataset, args.teacher, args.student) 
    else:
        log_name = '%s-%s-%s'%(args.dataset, args.teacher, args.student)
    
    if args.distributed:
        args.logger = [None] * args.world_size
        logger = args.logger[args.local_rank] = datafree.utils.logger.get_logger(log_name, output='checkpoints/datafree-%s/log-%s-%s-%s%s-R%d.txt'%(args.method, args.dataset, args.teacher, args.student, args.log_tag, args.local_rank))
        
    else:
        logger = args.logger = datafree.utils.logger.get_logger(log_name, output='checkpoints/datafree-%s/log-%s-%s-%s%s.txt'%(args.method, args.dataset, args.teacher, args.student, args.log_tag))
    if args.rank<=0:
        for k, v in datafree.utils.flatten_dict( vars(args) ).items(): # print args
            logger.info( "%s: %s"%(k,v) )

    ############################################
    # Setup dataset
    ############################################
    num_classes, ori_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    evaluator = datafree.evaluators.classification_evaluator(val_loader)
    loyalty_measurer = datafree.evaluators.prediction_agreement_evaluator(val_loader)

    ############################################
    # Setup models
    ############################################
    def prepare_model(model):
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
            return model
        elif args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                # print(ngpus_per_node)
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                return model
            else:
                model.to(args.local_rank)
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank], output_device=args.local_rank)
                return model
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            return model
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
            return model
    # print(args.dataset)
    if args.dataset == 'imagenet' or args.dataset == 'tiny_imagenet':
        if args.teacher.startswith('resnet'):
            args.teacher = args.teacher + '_imagenet'
        if args.student.startswith('resnet'):
            args.student = args.student + '_imagenet'
    student = registry.get_model(args.student, num_classes=num_classes)
    teacher = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
    if args.dataset == 'tiny_imagenet':
        teacher.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = teacher.fc.in_features
        teacher.fc = nn.Linear(num_ftrs, 200)
        teacher.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        teacher.maxpool = nn.Sequential()
        student.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = student.fc.in_features
        student.fc = nn.Linear(num_ftrs, 200)
        student.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        student.maxpool = nn.Sequential()
    args.normalizer = normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    # teacher.load_state_dict(torch.load('checkpoints/scratch/%s_%s.pth'%(args.dataset, args.teacher), map_location='cpu')['state_dict'])
    if args.nt2_mode == '01':
        folder = 'scratch_nt2_1'
    elif args.nt2_mode == '02':
        folder = 'scratch_nt2_2'
    elif args.nt2_mode == '005':
        folder = 'scratch_nt2_05'
    elif args.nt2_mode == 'noisy':
        folder = 'scratch_i'
    else:
        folder = 'scratch'
    
    ckpt = torch.load('checkpoints/%s/%s_%s.pth'%(folder, args.dataset, args.teacher), map_location='cpu')['state_dict']
    # if args.gpu == 0:
    #     print(ckpt.keys())
    # exit(-1)
    if args.dataset == 'tiny_imagenet' and not args.distributed:
        # print('************')
        ckpt = {'.'.join(k.split('.')[1:]):v for k,v in ckpt.items()}

    # print(args.method, args.distributed)
    # exit(-1)
    student = prepare_model(student)
    teacher = prepare_model(teacher)
    teacher.load_state_dict(ckpt)
    criterion = datafree.criterions.KLDiv()
    kd_steps = args.kd_steps
    g_step = args.g_steps
    if args.dataset.startswith('cifar') or args.dataset == 'svhn':
        img_size = 32
    elif args.dataset == 'tiny_imagenet':
        img_size = 64
    elif args.dataset == 'imagenet':
        img_size = 224
    else:
        raise ValueError('No such dataset as {}'.format(args.dataset))
    
    widen_factor = 1
    if args.teacher.startswith('wrn'):
        type = 'wider'
        widen_factor = int(args.teacher.split('_')[-1])
    else:
        type = 'normal'
    # print(args.batch_size)
    
    ############################################
    # Setup data-free synthesizers
    ############################################
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size
    
    if args.method=='deepinv':
        synthesizer = datafree.synthesis.DeepInvSyntheiszer(
                 teacher=teacher, student=student, num_classes=num_classes, 
                 img_size=(3, img_size, img_size), iterations=args.g_steps, lr_g=args.lr_g,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, tv=0.001, l2=0.0,
                 save_dir=args.save_dir, transform=ori_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu)
    elif args.method in ['zskt', 'dfad', 'dfq', 'dafl']:
        nz = 512 if args.method=='dafl' else 256
        # generator = datafree.models.generator.LargeGenerator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = datafree.models.generator.DCGAN_Generator_CIFAR10(nz=nz, ngf=64, nc=3, img_size=img_size, d=args.depth, cond=args.cond, type=type, widen_factor=widen_factor)
        if args.dataset == 'imagenet':
            generator = datafree.models.stylegan_network.Generator()
        # if args.dataset == 'tiny_imagenet':
        #     nz = 128
        #     generator = datafree.models.generator.TinyGenerator(z_dim=128, img_size=img_size)
        generator = prepare_model(generator)
        criterion = torch.nn.L1Loss() if args.loss=='l1' else datafree.criterions.KLDiv(T=args.T)
        synthesizer = datafree.synthesis.GenerativeSynthesizer(
                 teacher=teacher, student=student, generator=generator, nz=nz, 
                 img_size=(3, img_size, img_size), iterations=args.g_steps, lr_g=args.lr_g,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, act=args.act, balance=args.balance, criterion=criterion,
                 normalizer=args.normalizer, device=args.gpu)
    elif args.method=='cmi':
        nz = 256
        generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=img_size, nc=3)
        # generator = datafree.models.generator.DCGAN_Generator_CIFAR10(nz=nz, ngf=64, nc=3, img_size=img_size, d=args.depth, cond=args.cond, type=type, widen_factor=widen_factor)
        # if args.dataset == 'tiny_imagenet':
        #     nz = 128
        #     generator = datafree.models.generator.TinyGenerator(z_dim=128, img_size=img_size)
        generator = prepare_model(generator)
        feature_layers = None # use all conv layers
        if args.teacher=='resnet34' or args.teacher=='resnet50': # only use blocks
            feature_layers = [teacher.layer1, teacher.layer2, teacher.layer3, teacher.layer4]
        synthesizer = datafree.synthesis.CMISynthesizer(teacher, student, generator, 
                 nz=nz, num_classes=num_classes, img_size=(3, img_size, img_size), 
                 # if feature layers==None, all convolutional layers will be used by CMI.
                 feature_layers=feature_layers, bank_size=40960, n_neg=4096, head_dim=256, init_dataset=args.cmi_init,
                 iterations=args.g_steps, lr_g=args.lr_g, progressive_scale=False,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, cr=args.cr, cr_T=args.cr_T,
                 save_dir=args.save_dir, transform=ori_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu)
    elif args.method == 'zskd':
        
        synthesizer = datafree.synthesis.ZSKDSynthesis(
                 teacher=teacher, student=student, num_classes=num_classes, 
                 img_size=(3, 32, 32), iterations=args.g_steps, lr_g=args.lr_g,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
                 save_dir=args.save_dir, transform=ori_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu, 
                 T=args.T)
    elif args.method == 'dfme':
        nz = 256
        generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        criterion = torch.nn.L1Loss() if args.loss=='l1' else datafree.criterions.KLDiv(T=args.T)
        synthesizer = datafree.synthesis.DFMESynthesizer(
            teacher=teacher, student=student, generator=generator, nz=nz, img_size=(3, 32, 32), iterations=args.g_steps, lr_g=args.lr_g, synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,normalizer=args.normalizer, device=args.gpu, logit_correction=args.logit_correction, no_logit=args.no_logits, grad_epsilon=args.grad_epsilon, grad_m=args.grad_m,loss=args.loss
        )
    elif args.method == 'softtarget':
        synthesizer = datafree.synthesis.SoftTargetSynthesizer(
                 teacher=teacher, student=student, num_classes=num_classes, 
                 img_size=(3, 32, 32), iterations=args.g_steps, lr_g=args.lr_g,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
                 save_dir=args.save_dir, transform=ori_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu, a=args.act)

    elif args.method=='fast_meta':
        nz = 256
        generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        synthesizer = datafree.synthesis.FastMetaSynthesizer(teacher, student, generator,
                 nz=nz, num_classes=num_classes, img_size=(3, 32, 32), init_dataset=args.cmi_init,
                 save_dir=args.save_dir, device=args.gpu,
                 transform=ori_dataset.transform, normalizer=args.normalizer,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
                 iterations=args.g_steps, warmup=args.warmup, lr_g=args.lr_g, lr_z=args.lr_z,
                 adv=args.adv, bn=args.bn, oh=args.oh,
                 reset_l0=args.reset_l0, reset_bn=args.reset_bn,
                 bn_mmt=args.bn_mmt, is_maml=args.is_maml)

    elif args.method == 'cudfkd':
        G_list = []
        # E_list = []
        # L = teacher.num_blocks
        # for debug

        
        # img_size = 32 if args.dataset.startswith('cifar') else 64
        if args.curr_option == 'none':
            reduct = 'batchmean'
        else:
            reduct = 'none'
        
        #if args.loss == 'l1':
        #    criterion = torch.nn.L1Loss(reduction=reduct)
        #elif args.loss == 'l2':
        #    criterion = torch.nn.MSELoss(reduction=reduct)
        #else:
        #    criterion = datafree.criterions.KLDiv(reduction=reduct)
        if args.loss == 'l1':
            criterion = torch.nn.L1Loss()
        elif args.loss == 'l2':
            criterion = torch.nn.MSELoss()
        else:
            criterion = datafree.criterions.KLDiv(T=args.T)
        if args.no_feature:
            L = 1
        else:
            L = (1 + args.depth)
       
        assert args.no_feature or len(kd_steps) == L, 'error'
        for l in range(L):
            nz=512 if args.dataset.startswith('cifar') else 1024
            
            tg = datafree.models.generator.DCGAN_Generator_CIFAR10(nz=nz, ngf=64, nc=3, img_size=img_size, d=args.depth, cond=args.cond, type=type, widen_factor=widen_factor)
            # if args.dataset == 'tiny_imagenet':
            #     nz = 128
            #     tg = datafree.models.generator.TinyGenerator(z_dim=128, img_size=img_size)
            tg = prepare_model(tg)
            G_list.append(tg)
            # print(args.synthesis_batch_size)
            # E_list.append(E)
        synthesizer = datafree.synthesis.CuDFKDSynthesizer(
            teacher=teacher,
            student=student,
            G_list=G_list,
            nz=nz,
            num_classes=num_classes,
            img_size=img_size,
            iterations=g_step,
            lr_g=args.lr_g,
            synthesis_batch_size=args.synthesis_batch_size,
            sample_batch_size=args.batch_size,
            save_dir=args.save_dir,
            transform=ori_dataset.transform,
            normalizer=args.normalizer,
            device=args.gpu,
            lmda_ent=args.lmda_ent,
            adv=args.adv,
            oh=args.oh,
            act=args.act,
            adv_type=args.adv_type,
            bn=args.bn,
            T=args.T,
            memory=args.memory,
            distributed=args.distributed
        )
    elif args.method == 'adadfkd':

        if args.curr_option == 'none':
            reduct = 'batchmean'
        else:
            reduct = 'none'
        
        #if args.loss == 'l1':
        #    criterion = torch.nn.L1Loss(reduction=reduct)
        #elif args.loss == 'l2':
        #    criterion = torch.nn.MSELoss(reduction=reduct)
        #else:
        #    criterion = datafree.criterions.KLDiv(T=args.T, reduction=reduct)
        
        if args.loss == 'l1':
            criterion = torch.nn.L1Loss()
        elif args.loss == 'l2':
            criterion = torch.nn.MSELoss()
        else:
            criterion = datafree.criterions.KLDiv(T=args.T)

        nz=512 if args.dataset.startswith('cifar') else 1024
        
        tg = datafree.models.generator.DCGAN_Generator_CIFAR10(nz=nz, ngf=64, nc=3, img_size=img_size, d=args.depth, cond=args.cond, type=type, widen_factor=widen_factor)
        # if args.dataset == 'tiny_imagenet':
        #     nz = 128
        #     tg = datafree.models.generator.TinyGenerator(z_dim=128, img_size=img_size)
        tg = prepare_model(tg)
        # G_list.append(tg)
            # E_list.append(E)
        synthesizer = datafree.synthesis.AdaSynthesizer(
            teacher=teacher,
            student=student,
            G_list=[tg],
            nz=nz,
            num_classes=num_classes,
            img_size=img_size,
            iterations=g_step,
            lr_g=args.lr_g, 
            synthesis_batch_size=args.synthesis_batch_size,
            sample_batch_size=args.batch_size,
            save_dir=args.save_dir,
            transform=ori_dataset.transform,
            normalizer=args.normalizer,
            device=args.gpu,
            lmda_ent=args.lmda_ent,
            adv=args.adv,
            oh=args.oh,
            act=args.act,
            adv_type=args.adv_type,
            bn=args.bn,
            T=args.T,
            memory=args.memory,
            hard=args.hard,
            tau=args.tau,
            mu=args.mu,
            mode=args.mode,
            neg=args.neg,
            n_neg=args.N_neg,
            distributed=args.distributed,
            k=args.length
        )
    else: raise NotImplementedError
        
    ############################################
    # Setup optimizer
    ############################################
    optimizers = []
    
    optimizer = torch.optim.SGD(student.parameters(), args.lr, weight_decay=args.weight_decay, momentum=0.9)
    optimizers.append(optimizer)

    # milestones = [ int(ms) for ms in args.lr_decay_milestones.split(',') ]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=milestones, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=args.epochs)
    
    ############################################
    # Resume
    ############################################
    args.current_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            if isinstance(student, nn.Module):
                student.load_state_dict(checkpoint['state_dict'])
            else:
                student.module.load_state_dict(checkpoint['state_dict'])
            best_acc1 = checkpoint['best_acc1']
            try: 
                args.start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            except: print("Fails to load additional model information")
            print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                  .format(args.resume, checkpoint['epoch'], best_acc1))
        else:
            print("[!] no checkpoint found at '{}'".format(args.resume))
        
    ############################################
    # Evaluate
    ############################################
    if args.evaluate_only:
        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        print('[Eval] Acc={acc:.4f}'.format(acc=eval_results['Acc']))
        return

    ############################################
    # Train Loop
    ############################################
    global_iter = 0
    if args.method == 'cudfkd':
        g, v = torch.zeros(1), torch.zeros(args.batch_size)
    L = 1

    for epoch in range(args.start_epoch, args.epochs):
        args.current_epoch=epoch

        for k in range( args.ep_steps//kd_steps ):
            # print(k)
            # two-stage
            # 1. Data synthesis
            vis_result = None
            for l in range(L):
                if args.method != 'adadfkd':
                    vis_result = synthesizer.synthesize() # g_steps
                else:
                    warmup = epoch < int(args.begin_fraction * args.epochs)
                    hard_factor = min(max(0, (1 - args.length) * (epoch - int(args.begin_fraction * args.epochs)) / int(args.epochs * (args.end_fraction-args.begin_fraction))), 1 - args.length)
                    # print(hard_factor)
                    vis_result = synthesizer.synthesize(hard_factor=hard_factor, warmup=warmup)
                # 2. Knowledge distillation
                # kd_steps
                global_iter = train(synthesizer, [student, teacher], criterion, optimizer, args, kd_steps, l=l, global_iter=global_iter, save=(k==0), warmup=epoch<int(args.epochs * args.begin_fraction))
                if args.log_fidelity:
                    global_iter, avg_diff = global_iter

        if args.method == 'cudfkd' or args.method == 'adadfkd':
            synthesizer.adv = datafree.utils.get_alpha_adv(epoch, args, args.adv, type='linear')
            
        
        if vis_result is not None:
            for vis_name, vis_image in vis_result.items():
                if vis_image.shape[1] == 3:
                    datafree.utils.save_image_batch( vis_image, 'checkpoints/datafree-%s/%s%s.png'%(args.method, vis_name, args.log_tag) )
        
        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        if args.method.endswith('cudfkd') and (args.memory):
            
            synthesizer.data_pool.save_buffer()

        if args.log_fidelity:
            eval_f = loyalty_measurer(teacher, student, device=args.gpu)
            agreement, prob_loyalty = eval_f['agreement'], eval_f['prob_loyalty']

        logger.info('[Eval] Epoch={current_epoch} Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f} Lr={lr:.4f}'
                .format(current_epoch=args.current_epoch, acc1=acc1, acc5=acc5, loss=val_loss, lr=optimizer.param_groups[0]['lr']))
        if args.log_fidelity:
            info = '[Eval] Epoch={current_epoch} Agreement@1={agreement:.4f} Prob_loyalty={prob_loyalty:.4f} Generated_Difficulty={avg_diff:.4f}'.format(current_epoch=args.current_epoch, agreement=agreement, prob_loyalty=prob_loyalty, avg_diff=avg_diff)
            logger.info(info)
        scheduler.step()
        is_best = acc1 > best_acc1
        
        is_new_direct = (args.save_freq > 0 and (epoch + 1) % args.save_freq == 0)
        best_acc1 = max(acc1, best_acc1)
        if args.log_fidelity:
            best_agg1 = max(agreement, best_agg1)
            best_prob1 = max(prob_loyalty, best_prob1)
        if args.distributed:
            _best_ckpt = 'checkpoints/datafree-%s/%s-%s-%s-%s-R%d.pth'%(args.method, args.dataset, args.teacher, args.student, args.log_tag, args.local_rank)
        else:
            _best_ckpt = 'checkpoints/datafree-%s/%s-%s-%s-%s.pth'%(args.method, args.dataset, args.teacher, args.student, args.log_tag)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.local_rank % ngpus_per_node == 0):
            save_dict = {
                'epoch': epoch + 1,
                'arch': args.student,
                'state_dict': student.state_dict(),
                'best_acc1': float(best_acc1),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            if args.method == 'cudfkd' or args.method == 'adadfkd':
                save_dict['G'] = tg.state_dict()
                if args.method == 'adadfkd' and args.hard>0:
                    save_dict['neg_bank'] = synthesizer.neg_bank.queue
            save_checkpoint(save_dict, is_best, is_new_direct, epoch, _best_ckpt)
    if args.local_rank<=0 or args.distributed:
        logger.info("Best: %.4f"%best_acc1)
        if args.log_fidelity:
            logger.info("Best Agreement@1: %.4f\tBest Prob Loyalty: %.4f"%(best_agg1, best_prob1))


def train(synthesizer, model, criterion, optimizer, args, kd_step, l=0, global_iter=0, save=False, warmup=True):
    loss_metric = datafree.metrics.RunningLoss(datafree.criterions.KLDiv(reduction='sum'))
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1,5))
    student, teacher = model
    optimizer = optimizer
    student.train()
    teacher.eval()
    if args.distributed:
        logger = args.logger[args.local_rank]
    else:
        logger = args.logger
    history = (args.method == 'deepinv') or (args.method == 'cmi') or (args.method.endswith('cudfkd') and args.memory)
    for i in range(kd_step):
        # print(i)
        loss_s = 0.0
        if args.method == 'cudfkd':
            images = synthesizer.sample(l, args.memory)
        elif args.method == 'adadfkd':
            images = synthesizer.sample(l)
        else:
            images = synthesizer.sample()

        # print(images)
        
        
        if args.method == 'cudfkd' or args.method == 'adadfkd':
            if args.dataset == 'cifar10':
                alpha = 0.0001
            else:
                alpha = 0.00002
            lamda = datafree.datasets.utils.lambda_scheduler(args.lambda_0, global_iter, alpha=alpha)

        if l == 0 and not history:
            images = synthesizer.normalizer(images.detach())

        # print(images)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        with args.autocast():
            with torch.no_grad():
                t_out, t_feat = teacher(images, return_features=True)
            
            s_out, s_feat = student(images.detach(), return_features=True)
            
            loss_s = criterion(s_out, t_out.detach())
            # print(loss_s)
            # exit(-1)
            if args.method == 'adadfkd' and args.s_nce > 0:
                loss_t_feat, loss_infonce = difficulty_mining(t_feat=t_feat, s_feat=s_feat, tau=args.tau, device=args.gpu)

        avg_diff = 0
        if args.curr_option != 'none':
            #real_loss_s = loss_s.sum(1) if args.loss == 'kl' else loss_s.mean(1)
            real_loss_s = loss_s
            if args.s_nce > 0:
                real_loss_s += loss_infonce * args.s_nce
            
            if args.curr_option.startswith('curr'):
                with torch.no_grad():
                    g,v = datafree.datasets.utils.curr_v(l=real_loss_s, lamda=lamda, spl_type=args.curr_option.split('_')[1])

            loss_s = (v*real_loss_s).mean()
            # print(v.mean(), real_loss_s, loss_s)
            #loss_s = (v*real_loss_s).sum()
            avg_diff = (v * real_loss_s).sum() / v.sum()  
        optimizer.zero_grad()
        if args.fp16:
            scaler_s = args.scaler_s
            scaler_s.scale(loss_s).backward()
            scaler_s.step(optimizer)
            scaler_s.update()
        else:
            loss_s.backward()
            optimizer.step()
        acc_metric.update(s_out, t_out.max(1)[1])
        loss_metric.update(s_out, t_out)
        global_iter += 1
        if args.print_freq>0 and i % args.print_freq == 0:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            logger.info('[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
            .format(current_epoch=args.current_epoch, i=i, total_iters=kd_step, train_acc1=train_acc1, train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()
    
    # exit(-1)
    if args.log_fidelity:
        return global_iter, avg_diff
    else:
        return global_iter
    
def save_checkpoint(state, is_best, is_save_all, epoch=0, filename='checkpoint.pth'):
    if is_save_all:
        if not os.path.exists('{}_temp/'.format(filename[:-4])):
            os.mkdir('{}_temp/'.format(filename[:-4]))
        filename = '{}_temp/checkpoint_{}.pth'.format(filename[:-4], epoch)
    
    if is_best or is_save_all:
        
        torch.save(state, filename)

if __name__ == '__main__':
    main()
