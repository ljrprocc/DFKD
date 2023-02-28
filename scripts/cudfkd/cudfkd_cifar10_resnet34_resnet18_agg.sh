python datafree_kd.py \
--method cudfkd \
--dataset cifar10 \
--batch_size 512 \
--teacher resnet34 \
--student resnet18 \
--lr 0.1 \
--epochs 250 \
--kd_steps 10 \
--kd_steps_interval 10 \
--g_steps_interval 1 \
--ep_steps 400 \
--g_steps 1 \
--lr_g 0.001 \
--begin_fraction 0.25 \
--end_fraction 0.75 \
--grad_adv 0.3 \
--adv 0. \
--depth 2 \
--T 20 \
--lmda_ent -20 \
--oh 1 \
--act 0. \
--gpu 2 \
--seed 0 \
--bn 1 \
--save_dir run/cudfkd_retest \
--log_tag cudfkd_retest \
--data_root ~/cifar10/ \
--no_feature \
--adv_type kl \
--nt2_mode 02 \
--curr_option curr_log \
--lambda_0 2.35 \
--log_fidelity \
--loss kl
