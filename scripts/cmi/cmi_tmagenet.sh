for i in {1..10};
do k=`echo "1+$i*0.1"|bc`;
python datafree_kd.py \
--method cmi \
--dataset tiny_imagenet \
--batch_size 32 \
--synthesis_batch_size 64 \
--teacher resnet34 \
--student resnet18 \
--lr 0.1 \
--kd_steps 2000 \
--ep_steps 2000 \
--g_steps 200 \
--lr_g 1e-3 \
--epochs 40 \
--lr_decay_milestones 25,30,35 \
--adv 0.5 \
--bn 1 \
--oh 0.5 \
--cr 0.8 \
--cr_T 0.1 \
--T 20 \
--act 0 \
--balance 0 \
--gpu 4 \
--adv_type kl \
--log_fidelity \
--cmi_init run/cmi-preinverted-wrn402 \
--save_dir run/adv_cmi_tim1 \
--log_tag adv_cmi_timagenet_test \
--data_root ~/timagenet/tiny-imagenet-200/;
done