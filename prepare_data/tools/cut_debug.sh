# SOURCE_FOLDER='pic20gt/mask_tag/test'
# TARGET_FOLDER='cut_npy_debug'

# # 创建目标文件夹，如果它不存在的话
# mkdir -p "$TARGET_FOLDER"

# # 查找并复制所有的npy文件
# find "$SOURCE_FOLDER" -name '*.npy' -exec cp {} "$TARGET_FOLDER" \;

NPY_FOLDER='cut_npy_debug'
TARGET_FOLDER='cut_npy_debug_cut'
python tools/cut_satellite.py -dataset_path "$NPY_FOLDER" -o "$TARGET_FOLDER"  \
    --clip_size 1024 --stride_size 1024 --is_GT