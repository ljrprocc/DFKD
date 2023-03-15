python datafree_kd.py \
--method cmi \
--dataset cifar10 \
--batch_size 256 \
--teacher resnet34 \
--student resnet18 \
--epochs 250 \
--lr 0.1 \
--kd_steps 400 \
--ep_steps 400 \
--g_steps 200 \
--lr_g 1e-3 \
--adv 0.5 \
--bn 1.0 \
--oh 1.0 \
--cr 0.8 \
--cr_T 0.2 \
--act 0 \
--balance 0 \
--gpu 5 \
--seed 40 \
--T 20 \
--save_dir run/cmi_10_nt02_3 \
--data_root ~/cifar10 \
--log_fidelity \
--nt2_mode 02 \
--log_tag cmi02_3
