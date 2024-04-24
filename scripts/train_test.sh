export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

WORK_DIR=work_dirs/cut_pic4dilate_segnext_mscan-t_1xb16-adamw-160k_satellite-resize2048x2048-weight-1-20-seg
CONFIG=configs/segnext/segnext_mscan-t_1xb16-adamw-160k_satellite-2048x2048-weight-nosampler-1-20.py

bash tools/dist_train.sh \
     $CONFIG \
     8  --work-dir $WORK_DIR

bash tools/dist_test.sh $CONFIG \
 $WORK_DIR/iter_80000.pth 8 \
 --show-dir $WORK_DIR  \
 --cfg-options test_dataloader.batch_size=4 val_dataloader.batch_size=4