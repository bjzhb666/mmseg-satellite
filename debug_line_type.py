from PIL import Image
import multiprocessing

# 定义灰度级别到颜色的映射关系
gray_to_color = {
    0: (255, 0, 0),     # Red
    1: (0, 255, 0),     # Green
    2: (0, 0, 255),     # Blue
    3: (255, 255, 0),   # Yellow
    4: (255, 0, 255),   # Magenta
    5: (0, 255, 255),   # Cyan
    6: (128, 0, 0),     # Maroon
    7: (0, 128, 0),     # Green (dark)
    8: (0, 0, 128),     # Navy
    9: (0,0,0), # white
    10: (255, 255, 255),  # Olive 
    100: (128, 128, 128)  # Gray
}

# 检测work_dirs/three_seg_head中以gt_line_type开头的文件，保存为伪彩色图像的名字
import os
import cv2
def process_pixel(x, y, gray_value):
    color = gray_to_color.get(gray_value, (0, 0, 0))  # 默认黑色
    return (x, y, color)

def grayscale_to_color(gray_image_path, output_image_path):
    # 读取灰度图像
    gray_image = Image.open(gray_image_path).convert('L')  # 'L' 模式将图像转换为灰度图

    # 创建一个新的 RGB 图像对象
    color_image = Image.new('RGB', gray_image.size)

    # 并行处理像素级别的灰度映射
    pool = multiprocessing.Pool()
    results = []
    for x in range(gray_image.width):
        for y in range(gray_image.height):
            gray_value = gray_image.getpixel((x, y))
            results.append(pool.apply_async(process_pixel, args=(x, y, gray_value)))

    # 获取并填充颜色结果
    for result in results:
        x, y, color = result.get()
        color_image.putpixel((x, y), color)

    # 关闭进程池并等待所有任务完成
    pool.close()
    pool.join()

    # 保存生成的彩色图像
    color_image.save(output_image_path)

    print(f"彩色图像已保存至 {output_image_path}")


path = 'work_dirs/three_seg_head'
color_path = 'work_dirs/three_seg_head/color'
if not os.path.exists(color_path):
    os.makedirs(color_path)
files = os.listdir(path)
for i, file in enumerate(files):
    if file.startswith('gt_line_type'):
        gray_image = os.path.join(path, file)
        color_image_rgb = grayscale_to_color(gray_image, gray_image.replace('gt_line_type', 'colorgt_line_type').replace(path, color_path))
        pred_gray=file.replace('gt_line_type', 'pred_line_type')
        pred_gray=os.path.join(path, pred_gray)
        color_image_rgb = grayscale_to_color(pred_gray, pred_gray.replace('pred_line_type', 'colorpred_line_type').replace(path, color_path))
    if i == 100:
        break
