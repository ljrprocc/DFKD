CUDA_VISIBLE_DEVICES=4,5 python datafree_kd.py \
--method cudfkd \
--dataset tiny_imagenet \
--batch_size 128 \
--teacher resnet50 \
--student resnet18 \
--lr 0.1 \
--epochs 400 \
--kd_steps 10 \
--ep_steps 400 \
--g_steps 1 \
--lr_g 0.001 \
--begin_fraction 0.2 \
--end_fraction 0.75 \
--grad_adv 0.1 \
--adv 1. \
--depth 3 \
--T 5 \
--lmda_ent -20 \
--oh 1 \
--act 0. \
--seed 0 \
--bn 1 \
--save_dir run/cudfkd_test \
--log_tag cudfkd_L3_line12_new_50 \
--data_root ~/timagenet/tiny-imagenet-200/ \
--multiprocessing_distributed \
--world_size 2 \
--rank 0 \
--no_feature \
--adv_type kl \
--curr_option curr_log \
--lambda_0 1.5 \
--log_fidelity \
--loss l1
