export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
WORK_DIR=work_dirs/new20-small-40k-merge-wodir-segdecoder
CONFIG=configs/segnext/segnext_instance20_small.py

bash tools/dist_test.sh $CONFIG \
 $WORK_DIR/iter_40000.pth 8 \
    --cfg-options test_dataloader.batch_size=4 val_dataloader.batch_size=4  \
     default_hooks.logger.interval=1 test_evaluator.instance_dir=$WORK_DIR/instance_dir \
     default_hooks.visualization.draw=True \
     --out $WORK_DIR/output
   #  --show-dir $WORK_DIR
