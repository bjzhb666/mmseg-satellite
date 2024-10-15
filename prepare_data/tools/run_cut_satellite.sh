# cut_satellite.py需要输入的参数是-dataset_path和'-o', '--out_dir' 19级20级要用不同的clip_size和stride_size
# SOURCE_FOLDER='pic20gt'
# SOURCE_FOLDER='19picsgt'
# TARGET_FOLDER="$SOURCE_FOLDER-cut"
SOURCE_FOLDER=$1 # 存放切完train val test的文件夹
TARGET_FOLDER="$SOURCE_FOLDER-cut"
CLIP_SIZE=$3

# 获取所有直接子文件夹的名字，并循环输出
find "$SOURCE_FOLDER"/* -maxdepth 0 -type d | while read -r folder; do
    find "$folder"/* -maxdepth 0 -type d | while read -r subfolder; do
        echo "$subfolder"
        echo "$TARGET_FOLDER/$(basename "$folder")/$(basename "$subfolder")"
        python tools/cut_satellite.py -dataset_path "$subfolder" -o "$TARGET_FOLDER/$(basename "$folder")/$(basename "$subfolder")" \
            --is_GT --clip_size $CLIP_SIZE --stride_size $CLIP_SIZE
    done
    echo ""
done

echo "start cutting images"
echo ""

PIC_SOURCE=$2
PIC_TARGET="$PIC_SOURCE-cut"

# 获取所有直接子文件夹的名字，并循环输出
find "$PIC_SOURCE"/* -maxdepth 0 -type d | while read -r folder; do
    # 如果上面的循环没有输出，则说明没有子文件夹，则执行下面的代码

    # echo "***not find subfolder***"
    echo "$folder"
    echo "$PIC_TARGET/$(basename "$folder")"
    python tools/cut_satellite.py -dataset_path "$folder" -o "$PIC_TARGET/$(basename "$folder")" \
        --clip_size  $CLIP_SIZE --stride_size  $CLIP_SIZE
done
