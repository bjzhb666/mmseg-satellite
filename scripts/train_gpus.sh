export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# bash tools/dist_train.sh \
#      configs/HLT/HLT_mscan-t_1xb8-adamw-160k_satellite-2048x2048-weight-1-20-seg.py \
#      8  --work-dir work_dirs/cut_pic1dilate_HLT_mscan-t_1xb8-adamw-160k_satellite-2048x2048-weight-1-20-seg 
bash tools/dist_train.sh \
     configs/segnext/segnext_instance20.py \
     8  --work-dir work_dirs/0604smallerclassweightdilate5 \
     --cfg-options default_hooks.checkpoint.interval=20000  train_cfg.val_interval=100000