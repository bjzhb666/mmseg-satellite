export CUDA_VISIBLE_DEVICES=4
python tools/test.py configs/segnext/segnext_instance.py \
 work_dirs/3heads/iter_40000.pth \
 --show-dir work_dirs/3heads 