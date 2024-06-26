export CUDA_VISIBLE_DEVICES=7
python tools/test.py configs/segnext/segnext_instance.py \
 work_dirs/onlyadditionalhead-dilate3-0528/iter_80000.pth \
    --show-dir work_dirs/debug  \
    --cfg-options test_dataloader.batch_size=4 val_dataloader.batch_size=4 default_hooks.logger.interval=1