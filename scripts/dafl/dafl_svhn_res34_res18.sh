python datafree_kd.py \
--method dafl \
--dataset svhn \
--batch_size 1024 \
--teacher wrn40_2 \
--student wrn40_1 \
--lr 0.1 \
--epochs 200 \
--kd_steps 5 \
--ep_steps 400 \
--g_steps 1 \
--lr_g 1e-3 \
--adv 0 \
--T 20 \
--bn 0 \
--oh 1 \
--act 0.001 \
--balance 20 \
--gpu 4 \
--seed 0 \
--log_tag dafl_svhn_t4 \
--curr_option none \
--log_fidelity \
--data_root ~/svhn/
