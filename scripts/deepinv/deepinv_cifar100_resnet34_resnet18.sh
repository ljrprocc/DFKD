python datafree_kd.py --method deepinv --dataset cifar100 --batch_size 256 --teacher resnet34 --student wrn40_1 --lr 0.1 --epochs 400 --kd_steps 400 --ep_steps 400 --g_steps 1000 --lr_g 0.1 --adv 1 --bn 10 --oh 1 --T 20 --act 0 --balance 0 --gpu 1 --seed 0 --save_dir ~/run/deepinv_100d_varyd --log_tag deepinv_100d_varyd --curr_option none --log_fidelity --data_root ~/cifar100/