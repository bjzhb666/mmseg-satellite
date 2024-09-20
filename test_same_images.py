import os
from PIL import Image
import numpy as np

def compare_images(img1_path, img2_path):
    # 打开图片并转换为灰度模式
    img1 = Image.open(img1_path).convert('L')
    img2 = Image.open(img2_path).convert('L')

    # 比较图片尺寸
    if img1.size != img2.size:
        return False
    
    # 转换为numpy数组
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    
    # 比较图片像素值
    return not np.array_equal(img1_array, img2_array)

def find_different_pairs(directory):
    image_pairs = {}
    
    # 遍历文件夹中的文件
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            base_name = filename.rsplit('-', 1)[0]
            if base_name not in image_pairs:
                image_pairs[base_name] = []
            image_pairs[base_name].append(filename)
    
    different_pairs = []
    
    # 比较图片对
    for base_name, files in image_pairs.items():
        if len(files) == 2:
            img1_path = os.path.join(directory, files[0])
            img2_path = os.path.join(directory, files[1])
            if compare_images(img1_path, img2_path):
                different_pairs.append((files[0], files[1]))
    
    return different_pairs

# 使用例子
directory_path = 'work_dirs/debug_same_input'
different_pairs = find_different_pairs(directory_path)
if different_pairs:
    print("像素值不完全相同的图片对有：")
    for pair in different_pairs:
        print(pair)
else:
    print("所有图片对的像素值都相同。")


# ### 比较两个tensor是否一致
# import torch
# tensor1 = torch.load('x1.pt')
# tensor2 = torch.load('x2.pt')

# if torch.equal(tensor1, tensor2):
#     print("两个tensor完全一致")
# else:
#     print("两个tensor不一致")