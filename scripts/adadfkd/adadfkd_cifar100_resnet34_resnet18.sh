python datafree_kd.py \
--method adadfkd \
--dataset cifar100 \
--batch_size 768 \
--teacher resnet34 \
--student resnet18 \
--lr 0.1 \
--epochs 700 \
--kd_steps 5 \
--kd_steps_interval 10 \
--g_steps_interval 1 \
--ep_steps 400 \
--g_steps 1 \
--begin_fraction 0.2 \
--end_fraction 0.8 \
--grad_adv 0.0 \
--lr_g 0.001 \
--adv 0. \
--depth 2 \
--T 5 \
--lmda_ent -20 \
--oh 1 \
--act 0. \
--gpu 2 \
--seed 20 \
--bn 1 \
--save_dir run/infonce_res_exp6_retest5 \
--log_tag infonce_res_exp6_retest5 \
--data_root ~/cifar100/ \
--no_feature \
--adv_type kl \
--curr_option curr_log \
--lambda_0 1.2 \
--hard 0.1 \
--length 0.65 \
--tau 0.07 \
--neg 0.0 \
--s_nce 0.1 \
--log_fidelity \
--mode memory \
--N_neg 24576 \
--loss l1
