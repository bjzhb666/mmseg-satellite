#!/bin/bash

# 定义目标文件夹路径
# TARGET_FOLDER="pic20gt-cut/mask_tag"
TARGET_FOLDER="$1/mask_tag"

# 使用 find 命令检查所有子文件夹中是否存在 .npy 文件
if [ -n "$(find "$TARGET_FOLDER" -mindepth 1 -maxdepth 1 -type d -exec sh -c 'ls -1 "{}"/*.npy 2>/dev/null' \;)" ]; then
    echo "子文件夹中存在 .npy 文件"
    # 输出npy文件的名字
    find "$TARGET_FOLDER" -mindepth 1 -maxdepth 1 -type d -exec sh -c 'ls -1 "{}"/*.npy 2>/dev/null' \;
else
    echo "子文件夹中不存在 .npy 文件"
fi
