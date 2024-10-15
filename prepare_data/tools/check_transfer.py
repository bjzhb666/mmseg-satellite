# This file is used to check the line type of "停车位" and visulize the line on the image.
import json
import os
from PIL import Image, ImageDraw

# 加载JSON文件
with open('20level_json/result.json', 'r', encoding='utf-8') as f:
    annotations = json.load(f)

# 图片文件夹路径
image_folder = 'picuse20/use'

# 保存绘制结果的文件夹路径
output_folder = 'debug/stop_line'
os.makedirs(output_folder, exist_ok=True)

# 遍历每张图片的标注信息
for img_name, img_data in annotations.items():
    # 打开对应的图片
    img_path = os.path.join(image_folder, img_name)
    image = Image.open(img_path)
    draw = ImageDraw.Draw(image)

    # 遍历所有线段
    for line in img_data['lines']:
        if line['line_type'] == '停车位':
            # 画出停车位线段
            points = line['points']
            for i in range(len(points) - 1):
                draw.line((points[i][0], points[i][1], points[i+1][0], points[i+1][1]), fill='red', width=2)

    # 保存绘制结果
    output_path = os.path.join(output_folder, img_name[:-4] + '_stop_line' + img_name[-4:])
    image.save(output_path)
    # 保存原图
    image.save(os.path.join(output_folder, img_name))

print("所有停车位线段已绘制完成并保存。")
