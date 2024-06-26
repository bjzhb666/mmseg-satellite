export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash tools/dist_test.sh configs/segnext/segnext_instance.py \
 work_dirs/onlyadditionalhead-dilate3-0528/iter_80000.pth 8 \
    --show-dir work_dirs/debug  \
    --cfg-options test_dataloader.batch_size=1 val_dataloader.batch_size=1 default_hooks.logger.interval=5