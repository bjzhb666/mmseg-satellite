'''
文件夹里面有一些灰度图，我想把灰度图中像素值为5的地方改为3，给出代码，同时要考虑并行化处理，使用Python多进程
'''
import os
from PIL import Image
import numpy as np
import multiprocessing
from tqdm import tqdm
from functools import partial

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Change pixel values in grayscale images.')
    parser.add_argument('--folder_path', required=True, help='Path to the folder containing grayscale images')
    return parser.parse_args()

def process_image(filename, folder_path):
    file_path = os.path.join(folder_path, filename)
    img = Image.open(file_path)
    img_array = np.array(img)

    # 将像素值为5的地方改为3
    img_array[img_array == 5] = 3

    # 转换回图像并保存
    new_img = Image.fromarray(img_array)
    new_img.save(file_path)
    return filename  # 返回处理过的文件名，用于进度显示

def main():
    args = parse_args()

    folder_path = args.folder_path

    # 获取文件夹中的所有文件
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # 创建进程池
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # 使用多进程处理文件，并显示进度条
    partial_func = partial(process_image, folder_path=folder_path)
    
    # 包装函数以便显示进度条
    for _ in tqdm(pool.imap_unordered(partial_func, all_files), total=len(all_files)):
        pass

    # 关闭进程池
    pool.close()
    pool.join()

    print(f"Processed {len(all_files)} images in {folder_path}.")

if __name__ == '__main__':
    main()
