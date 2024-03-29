python datafree_kd.py \
--method softtarget \
--dataset cifar10 \
--batch_size 256 \
--teacher resnet34 \
--student resnet18 \
--lr 0.1 \
--epochs 2000 \
--kd_steps 400 \
--ep_steps 400 \
--g_steps 1000 \
--lr_g 0.1 \
--act 1. \
--gpu 6 \
--seed 0 \
--save_dir run/softtarget \
--log_tag softtarget \
--data_root ../cifar10/