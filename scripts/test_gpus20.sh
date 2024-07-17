export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash tools/dist_test.sh configs/segnext/segnext_instance20_small.py \
 work_dirs/new20-small-240k/iter_240000.pth 8 \
    --cfg-options test_dataloader.batch_size=4 val_dataloader.batch_size=4 default_hooks.logger.interval=1 \
    --show-dir work_dirs/new20-small-240k
