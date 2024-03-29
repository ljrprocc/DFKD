import argparse
import os
import random
import shutil
import time
import warnings
from datetime import timedelta

import registry
import datafree

import torch
import torch.nn as nn
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

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Basic
parser.add_argument('--data_root', default='data')
parser.add_argument('--teacher', default='wrn40_2')
parser.add_argument('--student', default='wrn16_1')
parser.add_argument('--dataset', default='cifar100')
parser.add_argument('--transfer_set', default='cifar10')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_decay_milestones', default="120,150,180", type=str,
                    help='milestones for learning rate decay')
parser.add_argument('--alpha', default=1.0, type=float, help='balance weights for soft targets')
parser.add_argument('--beta', default=0, type=float, help='balance weights for hard targets')
parser.add_argument('--T', default=1, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate_only', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# Device & FP16
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--node_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
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
parser.add_argument('--curr_option', type=str, default='spl')
parser.add_argument('--lambda_0', type=float, default=1.)
parser.add_argument('--open_ratio', action="store_true", help='ratio flag.')
parser.add_argument('--ratio', type=float, default=1.0, help='ratio for different confidence of samples. in [0.2, 1.0]')
best_acc1 = 0

def main():
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "6666"
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
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args), join=True)
    else:
        # Simply call main_worker function
        main_worker(args.local_rank, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    global best_acc1
    args.local_rank = gpu

    ############################################
    # GPU and FP16
    ############################################
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.node_rank == -1:
            args.node_rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.node_rank = args.node_rank * ngpus_per_node + gpu

        # print(args.node_rank)
        
        # os.environ["RANK"] = str(args.local_rank)
        # os.environ["WORLD_SIZE"] = str(args.world_size)
        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=gpu, timeout=timedelta(seconds=40))
        # dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
        #                         world_size=args.world_size, rank=args.rank)
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
    log_name = 'R%d-%s-%s-%s'%(args.local_rank, args.dataset, args.teacher, args.student) if args.distributed else '%s-%s-%s'%(args.dataset, args.teacher, args.student)
    if args.distributed:
        logger = datafree.utils.logger.get_logger(log_name, output='checkpoints/vanilla_kd/log-%s-%s-%s-%s%s-rank-%d.txt'%(args.dataset, args.transfer_set, args.teacher, args.student, args.log_tag, args.local_rank))
        print('Logging checkpoints/vanilla_kd/log-%s-%s-%s-%s%s-rank-%d.txt'%(args.dataset, args.transfer_set, args.teacher, args.student, args.log_tag, args.local_rank))
        # exit(-1)
    else:
        logger = datafree.utils.logger.get_logger(log_name, output='checkpoints/vanilla_kd/log-%s-%s-%s-%s%s.txt'%(args.dataset, args.transfer_set, args.teacher, args.student, args.log_tag))
    if args.local_rank<=0 or args.distributed:
        for k, v in datafree.utils.flatten_dict( vars(args) ).items(): # print args
            logger.info( "%s: %s"%(k,v) )

    ############################################
    # Setup dataset
    ############################################
    num_classes, ori_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root, ratio=args.open_ratio, teacher=args.teacher, ratio_num=args.ratio, gpu=args.gpu)
    if os.path.isdir(args.transfer_set):
        train_dataset = datafree.utils.UnlabeledImageDataset(args.transfer_set, transform=ori_dataset.transform)
        args.transfer_set = args.transfer_set.strip('/').replace('/', '-')
    else:
        _, train_dataset, _ = registry.get_dataset(name=args.transfer_set, data_root=args.data_root, ratio=args.open_ratio, teacher=args.teacher, ratio_num=args.ratio, gpu=args.gpu)
    # train_dataset.transforms = train_dataset.transform = ori_dataset.transform
    if args.local_rank <= 0:
        print(train_dataset)
    cudnn.benchmark = True
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    evaluator = datafree.evaluators.classification_evaluator(val_loader)

    ############################################
    # Setup models
    ############################################
    if args.dataset == 'imagenet' or args.dataset == 'tiny_imagenet':
        if args.teacher.startswith('resnet'):
            args.teacher = args.teacher + '_imagenet'
        if args.student.startswith('resnet'):
            args.student = args.student + '_imagenet'
    student = registry.get_model(args.student, num_classes=num_classes)
    teacher = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
    normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    args.normalizer = normalizer
    if args.dataset == 'imagenet':
        teacher.load_state_dict(torch.load('checkpoints/scratch/%s_%s.pth'%(args.dataset, args.teacher), map_location='cpu'))
    else:
        teacher.load_state_dict(torch.load('checkpoints/scratch/%s_%s.pth'%(args.dataset, args.teacher), map_location='cpu')['state_dict'])

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
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                return model
            else:
                model.to(args.local_rank)
                # args.batch_size = int(args.batch_size / ngpus_per_node)
                # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank])
                return model
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            return model
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
            return model
    student = prepare_model(student)
    teacher = prepare_model(teacher)
    
    ############################################
    # Setup optimizer
    ############################################
    #criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(student.parameters(), args.lr, weight_decay=args.weight_decay, momentum=0.9)
    milestones = [ int(ms) for ms in args.lr_decay_milestones.split(',') ]
    scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=milestones, gamma=0.1)
    # print('************** after scheduler **************')

    ############################################
    # Resume
    ############################################
    args.current_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
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
        eval_results = evaluator(student, device=args.local_rank)
        (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        print('[Eval] Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f}'.format(acc1=acc1, acc5=acc5, loss=val_loss))
        return

    ############################################
    # Train Loop
    ############################################
    for epoch in range(args.start_epoch, args.epochs):
        # print('**********epoch start*************')
        if args.distributed:
            train_sampler.set_epoch(epoch)
        args.current_epoch=epoch
        train( train_loader, [student, teacher], optimizer, epoch, logger, args)
        student.eval()
        eval_results = evaluator(student, device=args.local_rank if args.distributed else args.gpu)
        (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        logger.info('[Eval] Epoch={current_epoch} Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f} Lr={lr:.4f}'
                .format(current_epoch=args.current_epoch, acc1=acc1, acc5=acc5, loss=val_loss, lr=optimizer.param_groups[0]['lr']))
        scheduler.step()
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if args.distributed:
            _best_ckpt = 'checkpoints/vanilla_kd/%s-%s-%s-%s-R%d.pth'%(args.dataset, args.transfer_set, args.teacher, args.student, args.local_rank)
        else:
            _best_ckpt = 'checkpoints/vanilla_kd/%s-%s-%s-%s.pth'%(args.dataset, args.transfer_set, args.teacher, args.student)
        if not args.multiprocessing_distributed or (args.distributed
                and args.local_rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.student,
                'state_dict': student.state_dict(),
                'best_acc1': float(best_acc1),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, _best_ckpt)
    if args.local_rank<=0 or args.distributed:
        logger.info("Best: %.4f"%best_acc1)

def train(train_loader, model, optimizer, epoch, logger, args):
    loss_metric = datafree.metrics.RunningLoss(datafree.criterions.KLDiv(reduction='sum'))
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1,5))
    student, teacher = model
    optimizer = optimizer
    student.train()
    teacher.eval()
    for i, data in enumerate(train_loader):
        # print('*************** begin training **************')
        global_iter = epoch * len(train_loader) + i
        lamda = datafree.datasets.utils.lambda_scheduler(args.lambda_0, global_iter)
        if isinstance(data, (tuple,list)):
            images, targets = data
        else:
            images, targets = data, data.new_zeros(len(data))
        
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
        else:
            images = images.to(args.local_rank, non_blocking=True)
            targets = targets.to(args.local_rank, non_blocking=True)

        with args.autocast():
            with torch.no_grad():
                t_out, t_feat = teacher(images, return_features=True)
            if args.curr_option == 'none':
                reduct = 'batchmean'
            else:
                reduct = 'none'
            s_out = student(images.detach())
            loss_kld = datafree.criterions.kldiv(s_out, t_out.detach(), T=args.T, reduction=reduct) * args.alpha
            if reduct == 'none':
                loss_kld = loss_kld.mean(1)
            if args.beta>0:
                loss_ce = torch.nn.functional.cross_entropy( s_out, targets, reduction=reduct) * args.beta
            else:
                loss_ce = 0
            loss_s = loss_kld + loss_ce

        if reduct == 'none':
            g,v = datafree.datasets.utils.curr_v(l=loss_s, lamda=lamda, spl_type=args.curr_option.split('_')[1])
            # print(loss_s.mean(), v.mean())
            # exit(-1)
            loss_s = (v * loss_s).sum() + g
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
        # print(i)
        if args.print_freq>0 and i % args.print_freq == 0:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            logger.info('[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
              .format(current_epoch=args.current_epoch, i=i, total_iters=len(train_loader), train_acc1=train_acc1, train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()

    if args.distributed:
        datafree.utils.save_image_batch( args.normalizer(images, True), 'checkpoints/vanilla_kd/data-R%d.png'%(args.local_rank) )
    else:
        datafree.utils.save_image_batch( args.normalizer(images, True), 'checkpoints/vanilla_kd/data-gpu-%d.png'%(args.gpu) )


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


if __name__ == '__main__':
    main()