python datafree_kd.py \
--method improved_cudfkd \
--dataset cifar100 \
--batch_size 1024 \
--teacher wrn40_2 \
--student wrn16_2 \
--lr 0.1 \
--epochs 400 \
--kd_steps 5 \
--kd_steps_interval 10 \
--g_steps_interval 1 \
--ep_steps 400 \
--g_steps 1 \
--begin_fraction 0.2 \
--end_fraction 0.8 \
--grad_adv 0.0 \
--lr_g 0.001 \
--adv 1.0 \
--depth 2 \
--T 2 \
--lmda_ent -20 \
--oh 1 \
--act 0. \
--gpu 3 \
--seed 0 \
--bn 1 \
--save_dir run/infonce_wrn162_exp1_100 \
--log_tag infonce_wrn162_exp1_100 \
--data_root ~/cifar100/ \
--no_feature \
--adv_type kl \
--curr_option curr_log \
--lambda_0 1.2 \
--hard 0.1 \
--length 0.9 \
--tau 0.07 \
--neg 0.0 \
--s_nce 0.05 \
--log_fidelity \
--N_neg 49152 \
--loss kl