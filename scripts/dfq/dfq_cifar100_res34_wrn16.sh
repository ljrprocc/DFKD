python datafree_kd.py \
--method dfq \
--dataset cifar100 \
--batch_size 256 \
--teacher resnet34 \
--student wrn40_1 \
--lr 0.1 \
--epochs 400 \
--kd_steps 10 \
--ep_steps 400 \
--g_steps 1 \
--lr_g 1e-3 \
--adv 1 \
--T 5 \
--bn 1 \
--oh 1 \
--act 0 \
--balance 20 \
--data_root ~/cifar100/ \
--gpu 1 \
--loss kl \
--log_tag dfq_100_vary2 \
--log_fidelity \
--seed 0 \