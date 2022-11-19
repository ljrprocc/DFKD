python datafree_kd.py --method adadfkd --dataset cifar10 --batch_size 1024 --teacher wrn40_2 --student wrn16_1 --lr 0.1 --epochs 300 --kd_steps 10 --kd_steps_interval 10 --g_steps_interval 1  --ep_steps 400 --g_steps 1 --lr_g 0.001 --begin_fraction 0.0 --end_fraction 0.75 --grad_adv 0.0 --adv 1. --depth 2 --T 20 --lmda_ent -20 --oh 1 --act 0. --gpu 5 --seed 0 --bn 1 --save_dir run/infonce_wrn_8192_length_test_a --log_tag infonce_wrn_8192_length_test_a --data_root ~/cifar10/ --no_feature --adv_type kl --curr_option curr_log --lambda_0 1.5 --log_fidelity --mode memory --hard 0.1 --tau 0.07 --s_nce 0.05 --length 0.9 --loss kl --N_neg 8192