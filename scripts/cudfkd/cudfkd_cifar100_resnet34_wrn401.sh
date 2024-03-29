for i in {1..10};
do k=`echo "0.05+$i*0.01"|bc`;
python datafree_kd.py \
--method cudfkd \
--dataset cifar100 \
--batch_size 1024 \
--teacher vgg11 \
--student wrn40_1 \
--lr 0.1 \
--epochs 400 \
--kd_steps 10 \
--ep_steps 400 \
--g_steps 1 \
--lr_g 0.001 \
--adv 0.1 \
--depth 2 \
--T 2 \
--lmda_ent -20 \
--oh 1 \
--act 0. \
--gpu 1 \
--seed 0 \
--bn 1 \
--save_dir run/cudfkd_test \
--log_tag cudfkd_retest_100$i \
--begin_fraction 0.2 \
--end_fraction 0.75 \
--grad_adv $k \
--data_root ~/cifar100/ \
--no_feature \
--adv_type kl \
--curr_option curr_log \
--lambda_0 1.5 \
--log_fidelity \
--loss kl;
done