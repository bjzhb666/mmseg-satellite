'''
所有ignorevalue都被设成了100，
这是因为在生成GT的时候，我们将访问次数大于1的点标记为了True，也就是ignore value
'''
import cv2
import numpy as np
import json
import os
from PIL import Image

LINE_WIDTH = 1
VISUALIZATION = False
IMAGE_SIZE = 4096

def generate_line_mask_without_direction(coordinates, line_width, image_size=(IMAGE_SIZE, IMAGE_SIZE)):
    """
    生成线的掩码（二值掩码），不包含方向
    :param coordinates: 线的坐标点
    :param line_width: 线的宽度
    :param image_size: 图像大小
    :return: 线的掩码
    """
    # 如果坐标点少于2个，返回None
    if len(coordinates) < 2:
        return None
    
    # 创建一个空白图像作为掩码
    mask = np.zeros(image_size, dtype=np.uint8)
    
    # 将坐标点转换为整数
    coordinates = np.array(coordinates, dtype=np.int32)
    
    # 绘制连续线段
    cv2.polylines(mask, [coordinates], False, 255, thickness=line_width)
    
    # 找到线的内部点并返回
    indices = np.where(mask == 255)
    line_points = np.column_stack((indices[1], indices[0]))
    
    return line_points

def generate_pic_mask(coordinates, pixel_value=1):
    """
    生成整张图片的掩码，一个instance生成一个掩码
    :param coordinates: 线的坐标点
    :param pixel_value: 线的像素值
    :param too_long: 如果在生成tag的时候，tag数量超过255，那么就需要创建full_mask的dtype为uint16
    :return: 图片的掩码, size=IMAGE_SIZE,IMAGE_SIZE
    """
    image_size = (IMAGE_SIZE, IMAGE_SIZE)
    # 创建一个空白图像作为完整掩码
    full_mask = np.zeros(image_size, dtype=np.uint8)

    # 将坐标点转换为整数
    coordinates = np.array(coordinates, dtype=np.int32)
    # 将坐标点超出图像范围的点删除
    coordinates = coordinates[coordinates[:, 0] < IMAGE_SIZE]
    coordinates = coordinates[coordinates[:, 1] < IMAGE_SIZE]
    # 将指定坐标点设为 pixel_value
    full_mask[coordinates[:, 1], coordinates[:, 0]] = pixel_value

    return full_mask


def read_json(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    return data


line_cls2id = {
    "车道线": 1,
    "道路边缘": 2,
    "虚拟线": 3,
}

color_cls2id = {
    "白": 1,
    "黄": 2,
    "其他": 3,
    "无": 4,
}

linetype_cls2id = {
    '导流区': 1, 
    '实线': 2, 
    '虚线': 3, 
    '停车位': 4, 
    '短粗虚线': 5, 
    '粗实线': 6, 
    '其他': 7, 
    '待转区': 8, 
    '引导线': 9,
    '无': 10,
}

# linenum_cls2id = {
#     '单线': 1, 
#     '双线': 2, 
#     '其他': 3, 
#     '无': 4,
# } # Zoom=19
linenum_cls2id = {'单线': 1, '双线': 2, '其他': 3, '无': 4, '四线':5} # zoom=20

attribute_cls2id = {'无': 1, '禁停网格': 2, '减速车道': 3, '公交车道': 4, 
                    '其他': 5, '潮汐车道': 6, '借道区': 7, '可变车道': 8}

direction_cls2id = {'无': 1, '双向': 2}

boundary_cls2id = {'是': 1, '否': 2}


def generate_gt(image_name, target_path, json_data, attr='category', idx=line_cls2id):
    """
        generate the semantic segmentation gt label for each image
    Args:
        image_name (str): 
        target_path (str): 
        json_data (dict): annotation json data
        attr (str): the attribute of the line, including 'category', 'color', 'line_type', 'num', 'attribute', 'direction', 'boundary'
    Returns:
        None
    """    
    
    png_name = image_name + ".png"
    pic_lines_list = json_data[png_name]["lines"]
    # print(pic_lines_list)
    len_lines = len(pic_lines_list)
    
    if len_lines == 0:
        # 返回一个全0的IMAGE_SIZE*IMAGE_SIZE的图片
        pic_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        cv2.imwrite(target_path, pic_mask)
        print("该图片没有标注")
        return

    # 创建一个标志映射，如果有经过两次或更多次的线点，那么这个坐标点就是无效的，将其标记为True
    visited = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for i in range(len_lines):
        line_key_points = pic_lines_list[i]["points"]
        line_points = generate_line_mask_without_direction(line_key_points, LINE_WIDTH)
        # 记录每个点的访问次数
        if line_points is None: # stupid annotation with the same point
            continue
        if len(line_points) == 0: # this instance is invalid (no points)
            # import pdb; pdb.set_trace()
            continue

        for point in line_points:
            # import pdb; pdb.set_trace()
            x, y = point
            visited[y, x] += 1 
            # 注意，这里我们将列索引放在前面，行索引放在后面，
            # 这是因为在图像处理中，通常先指定列（即x坐标），再指定行（即y坐标）。
            # 这与一般的矩阵索引（先行后列）是相反的。
    # 将visited中访问次数大于1的点标记为True
    visited = visited > 1

    
    # 开始生成GT mask
    for i in range(len_lines):
        
        line_cls = pic_lines_list[i][attr]
        # if line_cls == '虚拟线':
        #     print(image_name)
        #     import pdb; pdb.set_trace()
        line_key_points = pic_lines_list[i]["points"]  # 一个实例的points
        line_points = generate_line_mask_without_direction(line_key_points, LINE_WIDTH)

        if line_points is None:
            continue
        
        # 生成掩码
        if i == 0:
            pic_mask = generate_pic_mask(line_points, pixel_value=idx[line_cls])
        else:
            pic_mask = pic_mask + generate_pic_mask(
                line_points, pixel_value=idx[line_cls]
            ) # 假设一个是1，一个是2，相加就是3，但是并没有被排除

    # 去除掉visited中访问次数大于1的点
    pic_mask[visited] = 100  # 100是一个不可能的值，用于标记错误，也就是ignore value
    
    # import pdb; pdb.set_trace()
    if VISUALIZATION: # cannot visualize 3-channel image, meaningless
        # 可视化
        pic_mask = visualize_seg_map(pic_mask)
        # 保存可视化图片
        pic_mask.save(target_path)
    else: 
        cv2.imwrite(target_path, pic_mask)
       

# 可视化分割图函数
def visualize_seg_map(img):
    """
    Visualizes a segmentation map.
    Args:
        img (np.ndarray): The segmentation map to visualize.
    Returns:
        PIL.Image: Visualized segmentation map.
    """

    ignore_value = 100

    # Define the colors for all labels except the ignore value.
    color_map = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}

    # Create an empty colored image
    colored_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Color the image
    for k in color_map.keys():
        colored_img[img == k] = color_map[k]

    # Set the ignore value regions to white
    colored_img[img == ignore_value] = [255, 255, 255]

    # Convert colored numpy array back to PIL Image
    img_pil = Image.fromarray(colored_img)

    return img_pil


if __name__ == "__main__":
    # 读取json文件
    file_name = "20level_json/result.json"
    data = read_json(file_name)  # 是一个字典

    attr_list = ['category', 'color', 'line_type', 'num', 'attribute', 'direction', 'boundary']
    # 定义源文件夹和目标文件夹路径
    cls_id_list = [line_cls2id, color_cls2id, linetype_cls2id, linenum_cls2id, attribute_cls2id,
                    direction_cls2id, boundary_cls2id]
    # attr_list = ['category']
    # cls_id_list = [line_cls2id]
    
    for attr in attr_list:
        idx = cls_id_list[attr_list.index(attr)]
        print("正在处理{}属性".format(attr))
        source_folder = "picuse20"
        target_folder = 'pic20gt-' + attr

        i = 0
        # 遍历源文件夹下的所有子文件夹
        for root, dirs, files in os.walk(source_folder):
            # 遍历每个子文件夹中的图片文件
            for file in files:
                # 获取图片文件的路径和文件名
                image_path = os.path.join(root, file)
                image_name = os.path.splitext(file)[0]

                # 获取子文件夹相对于源文件夹的路径
                relative_path = os.path.relpath(root, source_folder)

                # 构建目标文件夹路径
                target_subfolder = os.path.join(target_folder, relative_path)

                # 如果目标子文件夹不存在，则创建它
                if not os.path.exists(target_subfolder):
                    os.makedirs(target_subfolder)

                # 构建目标文件路径
                target_path = os.path.join(target_subfolder, image_name + "-GT.png")

                # 进行图片生成GT标签的操作，这里仅做示例，你需要根据实际需求编写代码
                # 这里使用了shutil来复制源文件到目标文件夹
                # shutil.copy(image_path, target_path)
                generate_gt(image_name, target_path, data, attr=attr, idx=idx)
                # 输出操作信息，可根据需要注释掉
                # print(f'Generated GT label for {image_name}. Saved to {target_path}')
                
                i += 1
                if i % 100 == 0:
                    print("已处理{}张图片".format(i))

