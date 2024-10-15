# import os
# import shutil
# import argparse


# moved_count = 0  # 移动文件的计数器

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Convert satellite dataset to mmsegmentation format')
#     parser.add_argument('--txt_file_path', help='satellite folder path', )
#     parser.add_argument('--source_folder_path', help='output path',)
#     parser.add_argument('--target_folder_path', help='output path',)
 
    
#     args = parser.parse_args()
#     return args

# args = parse_args()
# # 请将下面的路径替换为实际的文件路径
# text_file_path = args.txt_file_path     # 文本文件路径
# source_folder_path = args.source_folder_path  # 源文件夹路径
# target_folder_path = args.target_folder_path  # 目标文件夹路径

# # 创建target_folder_path文件夹
# if not os.path.exists(target_folder_path):
#     os.makedirs(target_folder_path)

# # 打开文本文件并逐行读取
# with open(text_file_path, 'r') as file:
#     for line in file:
#         # import pdb; pdb.set_trace()
#         # 去除文件名的前后空格和换行符
#         filename = line.strip()
#         # 构建源文件路径
#         source_file_path_GT = os.path.join(source_folder_path, filename + '-GT.png')
#         source_file_path = os.path.join(source_folder_path, filename + '.png')
#         source_file_path_direction = os.path.join(source_folder_path, filename + '-GT-direction.png')

#         # 检查文件是否存在，存在哪种类型的文件就移动哪种类型的文件
#         if os.path.isfile(source_file_path):
#             # 移动文件到目标文件夹
#             shutil.move(source_file_path, target_folder_path)
#             moved_count += 1
#         elif os.path.isfile(source_file_path_GT):
#             shutil.move(source_file_path_GT, target_folder_path)
#             moved_count += 1
#         elif os.path.isfile(source_file_path_direction):
#             shutil.move(source_file_path_direction, target_folder_path)
#             moved_count += 1

# # 打印移动文件的数量以确认操作
# print(f"共移动了 {moved_count} 个文件到 {target_folder_path} 文件夹。")

import os
import shutil
import argparse
import multiprocessing

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert satellite dataset to mmsegmentation format')
    parser.add_argument('--txt_file_path', help='satellite folder path', )
    parser.add_argument('--source_folder_path', help='output path',)
    parser.add_argument('--target_folder_path', help='output path',)
    
    args = parser.parse_args()
    return args

def move_file(line, source_folder_path, target_folder_path):
    filename = line.strip()
    source_file_path_GT = os.path.join(source_folder_path, filename + '-GT.png')
    source_file_path = os.path.join(source_folder_path, filename + '.png')
    source_file_path_direction = os.path.join(source_folder_path, filename + '-GT-direction.png')

    if os.path.isfile(source_file_path):
        shutil.move(source_file_path, target_folder_path)
        return 1
    elif os.path.isfile(source_file_path_GT):
        shutil.move(source_file_path_GT, target_folder_path)
        return 1
    elif os.path.isfile(source_file_path_direction):
        shutil.move(source_file_path_direction, target_folder_path)
        return 1
    return 0

def main():
    args = parse_args()
    text_file_path = args.txt_file_path
    source_folder_path = args.source_folder_path
    target_folder_path = args.target_folder_path

    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)

    with open(text_file_path, 'r') as file:
        lines = file.readlines()

    with multiprocessing.Pool() as pool:
        results = pool.starmap(move_file, [(line, source_folder_path, target_folder_path) for line in lines])

    moved_count = sum(results)
    print(f"共移动了 {moved_count} 个文件到 {target_folder_path} 文件夹。")

if __name__ == '__main__':
    main()
