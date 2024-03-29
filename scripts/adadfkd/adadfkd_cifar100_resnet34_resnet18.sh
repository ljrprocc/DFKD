python datafree_kd.py \
--method adadfkd \
--dataset cifar100 \
--batch_size 1024 \
--teacher vgg11 \
--student resnet18 \
--lr 0.1 \
--epochs 400 \
--kd_steps 10 \
--ep_steps 400 \
--g_steps 1 \
--begin_fraction 0.2 \
--end_fraction 0.8 \
--grad_adv 0.0 \
--lr_g 0.001 \
--adv 1. \
--depth 2 \
--T 5 \
--lmda_ent -20 \
--oh 1 \
--act 0. \
--gpu 0 \
--seed 20 \
--bn 1 \
--save_dir run/abl_line32 \
--log_tag abl_line32 \
--data_root ~/cifar100/ \
--no_feature \
--adv_type kl \
--curr_option curr_log \
--lambda_0 1.0 \
--hard 0.1 \
--length 0.8 \
--tau 0.07 \
--neg 0.0 \
--s_nce 0.1 \
--log_fidelity \
--mode memory \
--N_neg 12288 \
--loss l1
