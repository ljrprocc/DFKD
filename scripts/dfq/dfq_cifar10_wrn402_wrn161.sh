for i in {2..11};
do k=`echo "0.9+$i*0.1"|bc`;
python datafree_kd.py \
--method dfq \
--dataset cifar100 \
--batch_size 512 \
--teacher vgg11 \
--student wrn40_1 \
--lr 0.1 \
--epochs 300 \
--kd_steps 10 \
--ep_steps 400 \
--g_steps 1 \
--lr_g 1e-3 \
--adv 1 \
--T $k \
--loss kl \
--data_root ~/cifar100 \
--bn 1 \
--oh 1 \
--act 0 \
--balance 20 \
--log_fidelity \
--log_tag vary_dfq_retest3$i \
--gpu 2 \
--seed 0;
done