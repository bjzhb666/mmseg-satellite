export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash tools/dist_test.sh configs/segnext/segnext_mscan-t_1xb16-adamw-160k_satellite-2048x2048-weight-nosampler-1-20.py \
 work_dirs/cut_pic1dilate_segnext_mscan-t_1xb16-adamw-160k_satellite-resize2048x2048-weight-1-20-seg/iter_16000.pth 8 \
 --show-dir work_dirs/showGT/nodilate  \
 --cfg-options test_dataloader.batch_size=4 val_dataloader.batch_size=4