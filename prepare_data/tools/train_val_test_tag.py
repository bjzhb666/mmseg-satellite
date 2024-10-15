import os
import shutil
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--source_folder", type=str, help="source folder")
argparser.add_argument("--target_folder", type=str, help="target folder")
argparser.add_argument("--pic_folder", type=str, help="pic folder to find txt files")
argparser.add_argument("--zoom", type=str, help="zoom level")
args = argparser.parse_args()

# 源文件夹路径
source_folder = args.source_folder

# 目标文件夹路径
target_folder = os.path.join(args.target_folder, "mask_tag")


# 读取train_filenames.txt，val_filenames.txt和test_filenames.txt中的文件名
def read_filenames(file_path):
    with open(file_path, "r") as f:
        filenames = f.read().splitlines()
    return filenames


pic_folder = args.pic_folder

train_filenames = read_filenames(
    os.path.join(pic_folder, args.zoom + "train_filenames.txt")
)
val_filenames = read_filenames(
    os.path.join(pic_folder, args.zoom + "val_filenames.txt")
)
test_filenames = read_filenames(
    os.path.join(pic_folder, args.zoom + "test_filenames.txt")
)

# 创建train/val/test文件夹
for folder in ["train", "val", "test"]:
    os.makedirs(os.path.join(target_folder, folder), exist_ok=True)


# 定义函数将文件从源文件夹复制到目标文件夹
def copy_files(filenames, folder):
    for filename in filenames:
        filename = filename[:-4]  # 去掉文件名后缀
        # 构造源文件和目标文件路径
        source_file_path_png = os.path.join(source_folder, filename + "-GT.png")
        source_file_path_npy = os.path.join(source_folder, filename + "-GT.npy")
        target_file_path = os.path.join(target_folder, folder, filename)

        # 复制png文件到目标文件夹中
        if os.path.exists(source_file_path_png):
            shutil.copy(source_file_path_png, target_file_path + "-GT.png")

        # 复制npy文件到目标文件夹中
        elif os.path.exists(source_file_path_npy):
            shutil.copy(source_file_path_npy, target_file_path + "-GT.npy")

        # 如果源文件不存在，则打印一条消息
        else:
            print(f"File '{filename}' does not exist.")


# 分别将train/test/val的文件复制到相应的文件夹中
# The `copy_files` function takes a list of filenames as input along with a folder name. It then
# iterates over each filename in the list, constructs the source file paths by appending "-GT.png" and
# "-GT.npy" to the filename, and constructs the target file path within the target folder.
copy_files(train_filenames, "train")
copy_files(val_filenames, "val")
copy_files(test_filenames, "test")

print("Files copied successfully!")
