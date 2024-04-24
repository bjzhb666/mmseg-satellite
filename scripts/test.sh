export CUDA_VISIBLE_DEVICES=4
python tools/test.py configs/segnext/segnext_mscan-t_1xb16-adamw-160k_satellite-2048x2048-weight-nosampler-1-20.py \
 work_dirs/cut_pic1dilate_segnext_mscan-t_1xb16-adamw-160k_satellite-resize2048x2048-weight-1-20-seg/iter_16000.pth \
 --show-dir work_dirs/showGT/nodilate 