python datafree_kd.py \
--method deepinv \
--dataset cifar10 \
--batch_size 256 \
--teacher wrn40_2 \
--student wrn40_1 \
--lr 0.1 \
--epochs 250 \
--kd_steps 400 \
--ep_steps 400 \
--g_steps 1000 \
--lr_g 0.1 \
--adv 1 \
--bn 10 \
--oh 1 \
--T 20 \
--act 0 \
--balance 0 \
--gpu 0 \
--seed 0 \
--save_dir run/deepinv \
--log_tag deepinv \
--curr_option none \
--log_fidelity \
--data_root ~/cifar10/ \