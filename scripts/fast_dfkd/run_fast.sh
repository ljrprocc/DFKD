# python datafree_kd.py --batch_size 256 --lr 0.2 --kd_steps 400 --ep_steps 400 --adv 1.0 --bn 10.0 --oh 0.5 --act 0 --balance 0 --gpu 3 --T 20 --seed 0 --bn_mmt 0.9 --warmup 20 --epochs 250 --dataset cifar100 --method fast_meta --g_steps 2 --lr_z 0.015 --lr_g 5e-3 --teacher vgg11 --student resnet18 --is_maml 1 --reset_l0 1 --reset_bn 0 --save_dir run/vgg-2 --log_tag vgg-2_test1 --data_root ~/cifar10 --dataset cifar10 --log_fidelity

# python datafree_kd.py --batch_size 256 --lr 0.2 --kd_steps 400 --ep_steps 400 --adv 1.0 --bn 10.0 --oh 0.5 --act 0 --balance 0 --gpu 0 --T 20 --seed 0 --bn_mmt 0.9 --warmup 20 --epochs 220 --dataset cifar100 --method fast_meta --g_steps 5 --lr_z 0.01 --lr_g 3e-3 --teacher vgg11 --student resnet18 --is_maml 1 --reset_l0 1 --reset_bn 0 --save_dir run/vgg-5 --log_tag vgg-5

python datafree_kd.py --batch_size 256 --lr 0.2 --kd_steps 400 --ep_steps 400 --adv 1.0 --bn 10.0 --oh 0.5 --act 0 --balance 0 --gpu 5 --T 20 --seed 0 --bn_mmt 0.9 --warmup 20 --epochs 320 --dataset svhn --method fast_meta --g_steps 10 --lr_z 0.01 --lr_g 2e-3 --teacher resnet32x4 --student resnet8x4 --is_maml 1 --reset_l0 1 --reset_bn 0 --save_dir run/resnet32_svhn --log_tag resnet32-svhn_1 --data_root ~/svhn --log_fidelity


# python datafree_kd.py --batch_size 256 --lr 0.2 --kd_steps 400 --ep_steps 400 --adv 1.1 --bn 10.0 --oh 0.4 --act 0 --balance 0 --gpu 0 --T 20 --seed 0 --bn_mmt 0.9 --warmup 20 --epochs 220 --dataset cifar100 --method fast_meta --g_steps 2 --lr_z 0.015 --lr_g 5e-3 --is_maml 1 --reset_l0 1 --reset_bn 0 --save_dir run/wrn-2 --log_tag wrn-2

# python datafree_kd.py --batch_size 256 --lr 0.2 --kd_steps 400 --ep_steps 400 --adv 1.1 --bn 10.0 --oh 0.4 --act 0 --balance 0 --gpu 0 --T 20 --seed 0 --bn_mmt 0.9 --warmup 20 --epochs 220 --dataset cifar100 --method fast_meta --g_steps 5 --lr_z 0.01 --lr_g 3e-3 --is_maml 1 --reset_l0 1 --reset_bn 0 --save_dir run/wrn-5 --log_tag wrn-5

# python datafree_kd.py --batch_size 256 --lr 0.2 --kd_steps 400 --ep_steps 400 --adv 1.1 --bn 10.0 --oh 0.4 --act 0 --balance 0 --gpu 0 --T 20 --seed 0 --bn_mmt 0.9 --warmup 20 --epochs 220 --dataset cifar100 --method fast_meta --g_steps 10 --lr_z 0.01 --lr_g 2e-3 --is_maml 1 --reset_l0 1 --reset_bn 0 --save_dir run/wrn-10 --log_tag wrn-10



# python datafree_kd.py --batch_size 256 --lr 0.2 --kd_steps 400 --ep_steps 400 --adv 1.1 --bn 10.0 --oh 0.4 --act 0 --balance 0 --gpu 0 --T 20 --seed 0 --bn_mmt 0.9 --warmup 20 --epochs 220 --dataset cifar100 --method fast_meta --g_steps 2 --lr_z 0.015 --lr_g 5e-3 --student wrn40_1 --is_maml 1 --reset_l0 1 --reset_bn 0 --save_dir run/wrnS401-2 --log_tag wrnS401-2

# python datafree_kd.py --batch_size 256 --lr 0.2 --kd_steps 400 --ep_steps 400 --adv 1.1 --bn 10.0 --oh 0.4 --act 0 --balance 0 --gpu 0 --T 20 --seed 0 --bn_mmt 0.9 --warmup 20 --epochs 220 --dataset cifar100 --method fast_meta --g_steps 5 --lr_z 0.01 --lr_g 3e-3 --student wrn40_1 --is_maml 1 --reset_l0 1 --reset_bn 0 --save_dir run/wrnS401-5 --log_tag wrnS401-5

# python datafree_kd.py --batch_size 256 --lr 0.2 --kd_steps 400 --ep_steps 400 --adv 1.1 --bn 10.0 --oh 0.4 --act 0 --balance 0 --gpu 0 --T 20 --seed 0 --bn_mmt 0.9 --warmup 20 --epochs 220 --dataset cifar100 --method fast_meta --g_steps 10 --lr_z 0.01 --lr_g 2e-3 --student wrn40_1 --is_maml 1 --reset_l0 1 --reset_bn 0 --save_dir run/wrnS401-10 --log_tag wrnS401-10



# python datafree_kd.py --batch_size 256 --lr 0.2 --kd_steps 400 --ep_steps 400 --adv 1.1 --bn 10.0 --oh 0.4 --act 0 --balance 0 --gpu 0 --T 20 --seed 0 --bn_mmt 0.9 --warmup 20 --epochs 220 --dataset cifar100 --method fast_meta --g_steps 2 --lr_z 0.015 --lr_g 5e-3 --student wrn16_2 --is_maml 1 --reset_l0 1 --reset_bn 0 --save_dir run/wrnS162-2 --log_tag wrnS162-2

# python datafree_kd.py --batch_size 256 --lr 0.2 --kd_steps 400 --ep_steps 400 --adv 1.1 --bn 10.0 --oh 0.4 --act 0 --balance 0 --gpu 0 --T 20 --seed 0 --bn_mmt 0.9 --warmup 20 --epochs 220 --dataset cifar100 --method fast_meta --g_steps 5 --lr_z 0.01 --lr_g 3e-3 --student wrn16_2 --is_maml 1 --reset_l0 1 --reset_bn 0 --save_dir run/wrnS162-5 --log_tag wrnS162-5

# python datafree_kd.py --batch_size 256 --lr 0.2 --kd_steps 400 --ep_steps 400 --adv 1.1 --bn 10.0 --oh 0.4 --act 0 --balance 0 --gpu 0 --T 20 --seed 0 --bn_mmt 0.9 --warmup 20 --epochs 220 --dataset cifar100 --method fast_meta --g_steps 10 --lr_z 0.01 --lr_g 2e-3 --student wrn16_2 --is_maml 1 --reset_l0 1 --reset_bn 0 --save_dir run/wrnS162-10 --log_tag wrnS162-10