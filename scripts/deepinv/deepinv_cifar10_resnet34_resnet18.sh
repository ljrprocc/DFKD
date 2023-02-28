python datafree_kd.py \
--method deepinv \
--dataset cifar10 \
--batch_size 256 \
--teacher resnet34 \
--student resnet18 \
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
--gpu 5 \
--seed 40 \
--save_dir /data/lijingru/run/deepinv_mt_2_3 \
--log_tag deepinv_nt_2_3 \
--curr_option none \
--log_fidelity \
--nt2_mode 02 \
--data_root ~/cifar10/
