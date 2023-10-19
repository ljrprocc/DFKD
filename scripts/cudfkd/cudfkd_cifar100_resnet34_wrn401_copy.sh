for i in {1..10};
do k=`echo "1+$i*0.1"|bc`;
python datafree_kd.py \
--method cudfkd \
--dataset cifar100 \
--batch_size 768 \
--teacher resnet34 \
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
--gpu 4 \
--seed 0 \
--bn 1 \
--save_dir run/cudfkd_test \
--log_tag cudfkd_retest4_$i \
--begin_fraction 0.2 \
--end_fraction 0.75 \
--grad_adv 0.05 \
--data_root ~/cifar100/ \
--no_feature \
--adv_type kl \
--curr_option curr_log \
--lambda_0 $k \
--log_fidelity \
--loss kl;
done