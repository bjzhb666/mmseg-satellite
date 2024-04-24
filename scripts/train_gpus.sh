export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# bash tools/dist_train.sh \
#      configs/HLT/HLT_mscan-t_1xb8-adamw-160k_satellite-2048x2048-weight-1-20-seg.py \
#      8  --work-dir work_dirs/cut_pic1dilate_HLT_mscan-t_1xb8-adamw-160k_satellite-2048x2048-weight-1-20-seg 
bash tools/dist_train.sh \
     configs/segnext/segnext_mscan-t_1xb16-adamw-160k_satellite-2048x2048-weight-nosampler-1-20.py \
     8  --work-dir work_dirs/cut_pic4dilate_segnext_mscan-t_1xb16-adamw-160k_satellite-resize2048x2048-weight-1-20-seg