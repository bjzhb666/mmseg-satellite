import cv2
import os
import numpy as np

# 定义灰度级别（0-255）
gray_levels = np.arange(256)
pixel_counts = {level: 0 for level in gray_levels}

folder_path_list = ['/data/hongbo.zhao/mmsegmentation/data/satellite20/num/train', 
                    '/data/hongbo.zhao/mmsegmentation/data/satellite20/num/test', 
                    '/data/hongbo.zhao/mmsegmentation/data/satellite20/num/val']

# for folder_path in folder_path_list:
#     print(f"处理文件夹 {folder_path}")

#     # 遍历文件夹中的所有图像文件
#     for filename in os.listdir(folder_path):
#         # 只处理PNG、JPEG等图像文件
#         if filename.endswith('.png'):
#             # 读取图像文件
#             image_path = os.path.join(folder_path, filename)
#             image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#             # 统计像素值出现的次数
#             unique, counts = np.unique(image, return_counts=True)
#             for level, count in zip(unique, counts):
#                 pixel_counts[level] += count

#     # 打印每个灰度级别的像素值出现次数
#     for level, count in pixel_counts.items():
#         if count > 0:
#             print(f"灰度级别 {level}: 出现次数 {count}")
#     print("")

import os
from PIL import Image
import numpy as np
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

def count_pixels_in_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    pixel_values = np.array(image).flatten()
    return Counter(pixel_values)

def count_pixel_values(image_folder):
    pixel_counts = Counter()
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith((".png", ".jpg", ".jpeg"))]
    
    with ProcessPoolExecutor() as executor:
        results = executor.map(count_pixels_in_image, image_paths)
        
    for result in results:
        pixel_counts.update(result)
    
    return pixel_counts

for folder_path in folder_path_list:
    print(f"Processing folder {folder_path}")
    pixel_counts = count_pixel_values(folder_path)
    
    for pixel_value, count in pixel_counts.items():
        print(f"Pixel value {pixel_value}: {count} times")
    print("")
    # 用pixel_counts出现次数最多的除其他的，得到倍数
    # 找到像素值中出现次数最多的像素值和对应的次数
    most_common_pixel_value = max(pixel_counts, key=pixel_counts.get)
    most_common_count = pixel_counts[most_common_pixel_value]

    # 输出结果
    print(f"Most common pixel value: {most_common_pixel_value} (appears {most_common_count} times)")

    # 用出现次数最多的像素值除以其他像素值，得到倍数
    for pixel_value, count in pixel_counts.items():
        if pixel_value != most_common_pixel_value:
            multiplier = most_common_count / count
            print(f"{most_common_pixel_value} / {pixel_value} = {multiplier}")


