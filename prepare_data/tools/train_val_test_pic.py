"""
Cut the pictures in the source folder into training set, validation set and test set, 
and save the names of the pictures in the corresponding set to the file.
ratio: 6:2:2
"""

import os
import random
import shutil
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--zoom", type=str, help="zoom level")
argparser.add_argument("--source_folder", type=str, help="source folder")
argparser.add_argument("--target_folder", type=str, help="target folder")
args = argparser.parse_args()

ZOOM = args.zoom

# 设置随机数种子
random.seed(42)

# 源文件夹路径
source_folder = args.source_folder
target_folder = args.target_folder
# 训练集、验证集和测试集文件夹路径
train_folder = os.path.join(target_folder, "train")
val_folder = os.path.join(target_folder, "val")
test_folder = os.path.join(target_folder, "test")

# 如果文件夹不存在，则创建
for folder in [train_folder, val_folder, test_folder]:
    os.makedirs(folder, exist_ok=True)

# 获取源文件夹中所有PNG图片的文件名
png_files = [file for file in os.listdir(source_folder) if file.endswith(".png")]

# 打乱图片顺序
random.shuffle(png_files)

# 计算切分点
train_split = int(0.6 * len(png_files))
val_split = int(0.8 * len(png_files))

# 切分为训练集、验证集和测试集
train_files = png_files[:train_split]
val_files = png_files[train_split:val_split]
test_files = png_files[val_split:]

# 复制训练集图片到训练集文件夹，并记录图片名字
train_filenames = []
for file in train_files:
    shutil.copy(os.path.join(source_folder, file), os.path.join(train_folder, file))
    train_filenames.append(file)

# 复制验证集图片到验证集文件夹，并记录图片名字
val_filenames = []
for file in val_files:
    shutil.copy(os.path.join(source_folder, file), os.path.join(val_folder, file))
    val_filenames.append(file)

# 复制测试集图片到测试集文件夹，并记录图片名字
test_filenames = []
for file in test_files:
    shutil.copy(os.path.join(source_folder, file), os.path.join(test_folder, file))
    test_filenames.append(file)

# 保存训练集、验证集和测试集图片名字到文件
with open(os.path.join(target_folder, ZOOM + "train_filenames.txt"), "w") as f:
    f.write("\n".join(train_filenames))

with open(os.path.join(target_folder, ZOOM + "val_filenames.txt"), "w") as f:
    f.write("\n".join(val_filenames))

with open(os.path.join(target_folder, ZOOM + "test_filenames.txt"), "w") as f:
    f.write("\n".join(test_filenames))
