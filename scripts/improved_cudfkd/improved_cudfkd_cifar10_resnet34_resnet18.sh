python datafree_kd.py --method improved_cudfkd --dataset cifar10 --batch_size 768 --teacher resnet34 --student resnet18 --lr 0.1 --epochs 250 --kd_steps 10 --kd_steps_interval 10 --g_steps_interval 1 --ep_steps 400 --g_steps 1 --lr_g 0.001 --begin_fraction 0.25 --end_fraction 0.75 --adv 1.0 --grad_adv 0.0 --depth 2 --T 20 --lmda_ent -20 --oh 1 --act 0. --gpu 0 --seed 0 --bn 1 --save_dir run/infonce_s_exp1_noisy  --log_tag infonce_s_exp1_noisy --data_root ~/cifar10/ --no_feature --adv_type kl --curr_option curr_log --lambda_0 2.3 --log_fidelity --mode memory --hard 0.0 --length 0.8 --tau 0.07 --neg 0.0 --s_nce 0.2 --loss kl --noisy