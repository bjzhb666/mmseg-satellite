# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import math
import os
import os.path as osp
import tempfile
import zipfile
import glob
import multiprocessing
from functools import partial

import mmcv
import numpy as np
from mmengine.utils import ProgressBar, mkdir_or_exist
import pdb

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert satellite dataset to mmsegmentation format')
    parser.add_argument('-dataset_path', help='satellite folder path', )
    parser.add_argument('-o', '--out_dir', help='output path',)
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=512)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=512)
    parser.add_argument(
        "--to_label",
        action="store_true",
        help="convert to label image")
    parser.add_argument(
        "--is_npy",
        action="store_true",
        help="whether the image is a npy file")
    parser.add_argument(
        "--is_GT",
        action="store_true",
        help="whether the image is a ground truth image")
    
    args = parser.parse_args()
    return args


def map_tag_to_uint8(arr):
    """map tag to uint8.

    Args:
        arr (ndarray): The input array.
    Returns:
        ndarray: The output mapped array (1-255). dtype is uint8.
    """
    unique_elements = np.unique(arr)
    if len(unique_elements) > 255:
        raise ValueError("Unique elements exceed 255.")
    # print(unique_elements)
    # 创建一个映射，将唯一元素映射到0-255范围内的整数，因为0是背景，所以unique_elements中第一个是0
    unique_to_255 = {element: idx  for idx, element in enumerate(unique_elements)}
    # print(unique_to_255)
    # import pdb; pdb.set_trace()
    # 应用映射到数组
    mapped_array = np.vectorize(lambda x: unique_to_255[x])(arr)
    mapped_array = mapped_array.astype(np.uint8)
    return mapped_array


def clip_big_image(image_path, clip_save_dir, args, to_label=False, is_npy=False, is_GT=False):
    # Original image of Potsdam dataset is very large, thus pre-processing
    # of them is adopted. Given fixed clip size and stride size to generate
    # clipped image, the intersection　of width and height is determined.
    # For example, given one 5120 x 5120 original image, the clip size is
    # 512 and stride size is 256, thus it would generate 20x20 = 400 images
    # whose size are all 512x512.
    '''
    is_npy: bool, whether the image is a npy file
    is_GT: bool, whether the image is a ground truth image, only used to control the file name
    '''
    if is_npy:
        image = np.load(image_path)
    else:
        image = mmcv.imread(image_path) # return ndarray
    # pdb.set_trace()
    h, w, c = image.shape
    clip_size = args.clip_size
    stride_size = args.stride_size

    num_rows = math.ceil((h - clip_size) / stride_size) if math.ceil(
        (h - clip_size) /
        stride_size) * stride_size + clip_size >= h else math.ceil(
            (h - clip_size) / stride_size) + 1
    num_cols = math.ceil((w - clip_size) / stride_size) if math.ceil(
        (w - clip_size) /
        stride_size) * stride_size + clip_size >= w else math.ceil(
            (w - clip_size) / stride_size) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * clip_size
    ymin = y * clip_size

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + clip_size > w, w - xmin - clip_size,
                           np.zeros_like(xmin))
    ymin_offset = np.where(ymin + clip_size > h, h - ymin - clip_size,
                           np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + clip_size, w),
        np.minimum(ymin + clip_size, h)
    ],
                     axis=1)

    if to_label:
        color_map = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0],
                              [255, 255, 0], [0, 255, 0], [0, 255, 255],
                              [0, 0, 255]])
        flatten_v = np.matmul(
            image.reshape(-1, c),
            np.array([2, 3, 4]).reshape(3, 1))
        out = np.zeros_like(flatten_v)
        for idx, class_color in enumerate(color_map):
            value_idx = np.matmul(class_color,
                                  np.array([2, 3, 4]).reshape(3, 1))
            out[flatten_v == value_idx] = idx
        image = out.reshape(h, w)

    for box in boxes:
        start_x, start_y, end_x, end_y = box
        clipped_image = image[start_y:end_y,
                              start_x:end_x] if to_label else image[
                                  start_y:end_y, start_x:end_x, :]
        # import pdb; pdb.set_trace()
        pic_name = osp.basename(image_path)
        tag_num = len(np.unique(clipped_image[:, :, 1]))

        if is_npy:
            if tag_num < 255:
                # change tag from 1 to 255
                clipped_image[:, :, 1] = map_tag_to_uint8(clipped_image[:, :, 1])
            else:
                print(f'tag number is {tag_num} in {pic_name}, save as npy')
                
        
        if is_GT:
            pic_name = pic_name.split('-GT')[0]
            if tag_num < 255:
                mmcv.imwrite(
                clipped_image.astype(np.uint8),
                osp.join(
                    clip_save_dir,
                    f'{pic_name}_{start_x}_{start_y}_{end_x}_{end_y}-GT.png'))
            else:
                np.save(
                    osp.join(
                        clip_save_dir,
                        f'{pic_name}_{start_x}_{start_y}_{end_x}_{end_y}-GT.npy'),
                    clipped_image)
        else: # pic files, not GT files, save as png
            pic_name = pic_name.split('.')[0] # remove the extension
        
            mmcv.imwrite(
                    clipped_image.astype(np.uint8),
                    osp.join(
                        clip_save_dir,
                        f'{pic_name}_{start_x}_{start_y}_{end_x}_{end_y}.png'))
            

# def main():
#     args = parse_args()

#     dataset_path = args.dataset_path
#     if args.out_dir is None:
#         out_dir = osp.join('data', 'cut_satellite')
#     else:
#         out_dir = args.out_dir

#     print('Making directories...')

#     mkdir_or_exist(out_dir)
#     # 读取所有的png文件
#     png_list = glob.glob(os.path.join(dataset_path, '*.png'))
#     # pdb.set_trace()
#     npy_list = glob.glob(os.path.join(dataset_path, '*.npy'))

#     print(f'Number of png files: {len(png_list)}')
#     print(f'Number of npy files: {len(npy_list)}')
    
#     dst_dir = osp.join(out_dir)
    
#     for i, png in enumerate(png_list):
#         if i % 100 == 0:
#             print(f'Processing {i+1}/{len(png_list)}')
#         clip_big_image(png, dst_dir, args, to_label=False, is_npy=False, is_GT=args.is_GT)
    
#     for i, npy in enumerate(npy_list):
#         if i % 100 == 0:
#             print(f'Processing {i+1}/{len(npy_list)}')
#         clip_big_image(npy, dst_dir, args, to_label=False, is_npy=True, is_GT=args.is_GT)
    
#     print('*******************Done!************************')

def process_file(filepath, dst_dir, args, is_npy):
    clip_big_image(filepath, dst_dir, args, to_label=False, is_npy=is_npy, is_GT=args.is_GT)

def main():
    args = parse_args()

    dataset_path = args.dataset_path
    out_dir = args.out_dir if args.out_dir is not None else os.path.join('data', 'cut_satellite')

    print('Making directories...')
    mkdir_or_exist(out_dir)

    png_list = glob.glob(os.path.join(dataset_path, '*.png'))
    npy_list = glob.glob(os.path.join(dataset_path, '*.npy'))

    print(f'Number of png files: {len(png_list)}')
    print(f'Number of npy files: {len(npy_list)}')
    
    dst_dir = out_dir

    # 创建进程池
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    # pool = multiprocessing.Pool(processes=1)

    # 处理 PNG 文件
    png_partial_func = partial(process_file, dst_dir=dst_dir, args=args, is_npy=False)
    pool.map(png_partial_func, png_list)

    # 处理 NPY 文件
    npy_partial_func = partial(process_file, dst_dir=dst_dir, args=args, is_npy=True)
    pool.map(npy_partial_func, npy_list)

    # 关闭进程池
    pool.close()
    pool.join()

    print('*******************Done!************************')

if __name__ == '__main__':
    main()

'''
to_label为True和False时，保存的clipped_image的形状可能会不同。
当to_label为True时，clipped_image是一个二维数组，形状为(end_y-start_y, end_x-start_x)，每个元素是一个类别标签。
当to_label为False时，clipped_image是一个三维数组，形状为(end_y-start_y, end_x-start_x, c)，每个元素是一个颜色值。
这是因为当to_label为False时，代码没有进行颜色映射操作，所以clipped_image仍然是一个彩色图像。
'''