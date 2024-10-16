export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1
python tools/train.py \
    configs/segnext/segnext_instance.py \
    --work-dir work_dirs/debug \
    --cfg-option train_dataloader.batch_size=1 train_dataloader.num_workers=0 \
        val_dataloader.num_workers=0 \
        train_dataloader.persistent_workers=False \
        val_dataloader.persistent_workers=False \
        test_dataloader.persistent_workers=False # \
        # train_cfg.val_interval=50
