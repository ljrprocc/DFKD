CUDA_VISIBLE_DEVICES=4,5 python datafree_kd.py \
--method adadfkd \
--dataset tiny_imagenet \
--batch_size 256 \
--teacher resnet50 \
--student resnet18 \
--lr 0.1 \
--epochs 500 \
--kd_steps 10 \
--ep_steps 400 \
--g_steps 1 \
--lr_g 0.001 \
--begin_fraction 0.2 \
--end_fraction 0.75 \
--grad_adv 0.10 \
--adv 1. \
--depth 3 \
--T 5 \
--lmda_ent -20 \
--oh 1 \
--act 0. \
--seed 0 \
--hard 0.05 \
--s_nce 0.02 \
--length 0.81 \
--tau 0.07 \
--bn 1 \
--save_dir run/adadfkd_test \
--log_tag adadfkd_new3 \
--data_root ~/timagenet/tiny-imagenet-200/ \
--multiprocessing_distributed \
--world_size 2 \
--rank 0 \
--no_feature \
--adv_type kl \
--curr_option curr_log \
--log_fidelity \
--lambda_0 1.3 \
--N_neg 16384 \
--loss l1
