export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

WORK_DIR=work_dirs/new19-40k-merge-wodir-segdecoder
CONFIG=configs/segnext/segnext_instance19_small.py

bash tools/dist_train.sh \
     $CONFIG \
     8  --work-dir $WORK_DIR \
     --cfg-options default_hooks.checkpoint.interval=10000 # \
     # train_cfg.val_interval=50

bash tools/dist_test.sh $CONFIG \
 $WORK_DIR/iter_40000.pth 8 \
 --show-dir $WORK_DIR  \
 --cfg-options test_dataloader.batch_size=4 val_dataloader.batch_size=4 \
  test_evaluator.instance_dir=$WORK_DIR/instance_dir