import os
from PIL import Image

# 设置源文件夹和目标文件夹
source_folder = 'nusc1024seg'
dest_folder = 'nusc4096seg'

# 创建目标文件夹
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith('.png'):
        # 解析文件名中的信息
        parts = filename[:-4].split('_')
        name = '_'.join(parts[:-4])
        x1, y1, x2, y2 = [int(x) for x in parts[-4:]]

        # 创建目标图像
        target_image = Image.new('RGB', (4096, 4096))

        # 打开小图片并粘贴到目标图像中
        small_image = Image.open(os.path.join(source_folder, filename))
        target_image.paste(small_image, (x1, y1))

        # 保存目标图像
        target_filename = f'{name}.png'
        target_image.save(os.path.join(dest_folder, target_filename))
        print(f'Saved {target_filename}')