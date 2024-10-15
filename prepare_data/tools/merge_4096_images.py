import os
from PIL import Image
import re

def merge_images(folder_name):
    # 获取文件夹中所有图片文件
    image_files = [f for f in os.listdir(folder_name) if f.endswith('.png')]
    merged_images_list = []
    os.makedirs('merged_images', exist_ok=True)
    for loc in ['singapore-onenorth', 'boston-seapot', 'singapore-hollandvillage', 'singapore-queenstown']:
    # for loc in ['singapore-onenorth-test', 'boston-seaport-test', 'singapore-queenstown-test']:
        # 提取每个图片的位置坐标
        positions = []
        for image_file in image_files:
            match = re.search(loc + r'_(-?\d+)_(-?\d+)_sat', image_file)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                positions.append((x, y, image_file))
                # print(x, y, image_file)
        print(len(positions))
        # 根据位置坐标对图片进行排序
        positions.sort(key=lambda x: (x[0], x[1]))
        
        print(positions)
        # 获取图片尺寸
        first_image = Image.open(os.path.join(folder_name, positions[0][2]))
        width, height = first_image.size
        print(width, height)
        first_x, first_y = positions[0][0], positions[0][1]
        print(first_x, first_y)
        # 创建合并后的大图
        # max_x = max(pos[0] for pos in positions)
        # max_y = max(pos[1] for pos in positions)
        count_x = len(set(pos[0] for pos in positions))
        count_y = len(set(pos[1] for pos in positions))
        # print(max_x, max_y)
        print(count_x, count_y)
        merged_image = Image.new('RGB', (count_x * width, count_y * height))
        
        # 将图片合并到大图上
        for x, y, image_file in positions:
            image = Image.open(os.path.join(folder_name, image_file))
            merged_image.paste(image, ((x-first_x) * width, (y-first_y) * height))
        
        # 保存合并后的图片
        merged_image.save(f'merged_images/merged_image-{loc}-4096.png')
        merged_images_list.append(merged_image)
    
    
    return merged_images_list

# 调用函数并保存合并后的图片
merged_image = merge_images('nusc4096seg')
# merged_image.save('merged_image-boston-seaport.png')
