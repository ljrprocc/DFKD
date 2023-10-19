python datafree_kd.py \
--method adadfkd \
--dataset cifar100 \
--batch_size 1024 \
--teacher wrn40_2 \
--student wrn40_1 \
--lr 0.1 \
--epochs 500 \
--kd_steps 10 \
--ep_steps 400 \
--g_steps 1 \
--begin_fraction 0.2 \
--end_fraction 0.8 \
--grad_adv 0.095  \
--lr_g 0.001 \
--adv 0.0 \
--depth 2 \
--T 2 \
--lmda_ent -20 \
--oh 1 \
--act 0. \
--gpu 3 \
--seed 20 \
--bn 1 \
--save_dir run/infonce_wrn401_s \
--log_tag infonce_401_s \
--data_root ~/cifar100/ \
--no_feature \
--adv_type kl \
--curr_option curr_log \
--lambda_0 1.2 \
--hard 0.1 \
--length 0.71 \
--tau 0.07 \
--neg 0.0 \
--s_nce 0.15 \
--log_fidelity \
--N_neg 16384 \
--loss kl