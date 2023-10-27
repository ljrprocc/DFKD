for i in {1..10};
do k=`echo "0.7+$i*0.02"|bc`;
python datafree_kd.py \
--method adadfkd \
--dataset cifar100 \
--batch_size 1024 \
--teacher vgg11 \
--student wrn40_1 \
--lr 0.1 \
--epochs 500 \
--kd_steps 10 \
--ep_steps 400 \
--g_steps 1 \
--lr_g 0.001 \
--adv 1.0 \
--depth 2 \
--T 2 \
--lmda_ent -20 \
--oh 1 \
--act 0. \
--gpu 4 \
--seed 0 \
--bn 1 \
--save_dir run/cudfkd_test \
--log_tag adadfkd_retest2_$i \
--begin_fraction 0.2 \
--end_fraction 0.75 \
--grad_adv 0.0 \
--data_root ~/cifar100/ \
--no_feature \
--adv_type kl \
--curr_option curr_log \
--lambda_0 1.2 \
--hard 0.1 \
--length $k \
--tau 0.07 \
--s_nce 0.1 \
--log_fidelity \
--loss kl;
done