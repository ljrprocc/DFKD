CUDA_VISIBLE_DEVICES=0,1,2,3 python vanilla_kd.py --dataset imagenet --teacher resnet50 --student resnet18 -j 8 --data_root ~/ILSVRC2012_imgs/ --batch_size 512 --transfer_set imagenet --curr_option none --print_freq 20 --T 4 --world_size 4 --dist_backend nccl --multiprocessing_distributed --node_rank 0