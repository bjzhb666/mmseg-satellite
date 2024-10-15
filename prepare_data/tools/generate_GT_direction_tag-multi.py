import cv2
import numpy as np
import json
import os
from PIL import Image
import argparse

LINE_WIDTH = 1
VISUALIZATION = False


def create_mask(height, width, keypoint1, keypoint2):
    # 初始化一个全False的mask数组
    mask = np.zeros((height, width), dtype=bool)

    # 由于关键点是以(x, y)形式给出的，我们需要正确分配它们到行和列
    # 获取矩形的左上角点，即xmin与ymin
    top_left = (min(keypoint1[1], keypoint2[1]), min(keypoint1[0], keypoint2[0]))

    # 获取矩形的右下角点，即xmax与ymax
    bottom_right = (max(keypoint1[1], keypoint2[1]), max(keypoint1[0], keypoint2[0]))

    # 将矩形区域内的值设置为True
    mask[top_left[0] : bottom_right[0] + 1, top_left[1] : bottom_right[1] + 1] = True

    # 判断keypoint2是否超出图片范围
    if keypoint2[0] >= width:
        keypoint2[0] = width - 1
    if keypoint2[1] >= height:
        keypoint2[1] = height - 1

    # 结束点不包括在矩形内
    mask[keypoint2[1], keypoint2[0]] = False

    return mask


def create_point_mask(points, H, W):
    mask = np.zeros((H, W), dtype=bool)
    for point in points:
        x, y = point
        # 检查点是否在mask的尺寸范围内
        if 0 <= x < W and 0 <= y < H:
            mask[y, x] = True
    return mask


def generate_line_mask(coordinates, line_width, image_size):
    """
    对于每个实例生成线的掩码（二值掩码）
    :param coordinates: 线的坐标点
    :param line_width: 线的宽度
    :param image_size: 图像大小
    :return:
        线的掩码
        direction mask: shape=(H, W, 2), H是高度，W是宽度，2是xy两个方向的方向向量
    """
    # 如果坐标点少于2个，返回None
    if len(coordinates) < 2:
        return None, None

    H, W = image_size
    # 创建一个空白图像作为掩码
    mask = np.zeros(image_size, dtype=np.uint8)
    # 创建一个direction mask，包含xy两个方向的方向向量，形状为(H, W, 2)
    direction_mask = np.zeros((H, W, 2), dtype=np.float32)

    # 将坐标点转换为整数
    coordinates = np.array(coordinates, dtype=np.int32)

    # 使用numpy的unique函数删除重复的行，并通过返回索引来重建原始数组
    _, idx = np.unique(coordinates, axis=0, return_index=True)

    # 根据索引排序以返回唯一元素的原始顺序，这一步是可选的
    coordinates = coordinates[np.sort(idx)]

    # 绘制连续线段
    cv2.polylines(mask, [coordinates], False, 255, thickness=line_width)

    # 找到线的内部点并返回
    indices = np.where(mask == 255)
    line_points = np.column_stack((indices[1], indices[0]))

    # get line direction: x,y as a vector
    for i in range(len(coordinates) - 1):
        p1 = coordinates[i]
        p2 = coordinates[i + 1]
        direction = p2 - p1

        if np.linalg.norm(direction) == 0:
            # import pdb; pdb.set_trace()
            print("The direction vector is zero, labeling error!")
            continue

        # normalize the direction vector
        direction = direction / np.linalg.norm(direction)
        # direction_list.append(direction)
        range_mask = create_mask(H, W, p1, p2)
        line_point_mask = create_point_mask(line_points, H, W)

        final_mask = np.logical_and(range_mask, line_point_mask)

        # direction_mask的第一wei是x方向，第二维是y方向
        # FIXME: 这里的方向向量是反的，应该是先y后x，这里改以后mm读数据那块也要改，但我们没改，所以规定后面都是这个顺序
        direction_mask[final_mask, 0] = direction[1]
        direction_mask[final_mask, 1] = direction[0]

    return line_points, direction_mask


def generate_line_mask_without_direction(coordinates, line_width, image_size):
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


def generate_pic_mask(coordinates, IMAGE_SIZE, pixel_value=1, too_long=False):
    """
    生成整张图片的掩码，一个instance生成一个掩码
    :param coordinates: 线的坐标点
    :param pixel_value: 线的像素值
    :param too_long: 如果在生成tag的时候，tag数量超过255，那么就需要创建full_mask的dtype为uint16
    :return: 图片的掩码, size=IMAGE_SIZE,IMAGE_SIZE
    """
    image_size = (IMAGE_SIZE, IMAGE_SIZE)
    # 创建一个空白图像作为完整掩码
    if too_long:
        full_mask = np.zeros(image_size, dtype=np.uint16)
    else:
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


def generate_gt(image_name, target_path_tag, target_path_dir, json_data, IMAGE_SIZE):
    """
        generate the semantic segmentation gt label for each image
    Args:
        image_name (str):
        target_path_tag (str): path for saving the tag and mask (instance id and sementic classes)
        target_path_dir (str): path for saving the direction mask
        json_data (dict): annotation json data
    Returns:
        None
    """

    png_name = image_name + ".png"
    pic_lines_list = json_data[png_name]["lines"]
    # print(pic_lines_list)
    len_lines = len(pic_lines_list)

    if len_lines == 0:  # 新数据没有这种情况
        # 返回一个全0的IMAGE_SIZE*IMAGE_SIZE的图片
        pic_three_channel = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        # pic_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        target_path_direction = target_path_dir.replace("-GT.png", "-GT-direction.png")
        # cv2.imwrite(target_path, pic_three_channel)
        cv2.imwrite(target_path_direction, pic_three_channel)
        cv2.imwrite(target_path_tag, pic_three_channel)
        print("该图片没有标注")
        return

    # 创建一个标志映射，如果有经过两次或更多次的线点，那么这个坐标点就是无效的，将其标记为True
    visited = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for i in range(len_lines):
        line_key_points = pic_lines_list[i]["points"]
        line_points, _ = generate_line_mask(
            line_key_points, LINE_WIDTH, (IMAGE_SIZE, IMAGE_SIZE)
        )
        # 记录每个点的访问次数
        if line_points is None:  # stupid annotation with the same point
            continue
        if len(line_points) == 0:  # this instance is invalid (no points)
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

    pic_direction_mask = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE, 2), dtype=np.float32
    )  # 整张图片的方向掩码
    for i in range(len_lines):

        line_cls = pic_lines_list[i]["category"]
        # if line_cls == '虚拟线':
        #     print(image_name)
        #     import pdb; pdb.set_trace()
        line_key_points = pic_lines_list[i]["points"]  # 一个实例的points
        line_points, direction_mask = generate_line_mask(
            line_key_points, LINE_WIDTH, (IMAGE_SIZE, IMAGE_SIZE)
        )

        if line_points is None:
            continue

        too_long = len_lines > 255
        # 生成掩码
        if i == 0:
            pic_mask = generate_pic_mask(
                line_points, IMAGE_SIZE=IMAGE_SIZE, pixel_value=line_cls2id[line_cls]
            )
            pic_tag = generate_pic_mask(
                line_points, IMAGE_SIZE=IMAGE_SIZE, pixel_value=i + 1, too_long=too_long
            )  # instance tag starts from 1
            pic_direction_mask = direction_mask
        else:
            pic_mask = pic_mask + generate_pic_mask(
                line_points, IMAGE_SIZE=IMAGE_SIZE, pixel_value=line_cls2id[line_cls]
            )  # 假设一个是1，一个是2，相加就是3，但是并没有被排除
            pic_tag = pic_tag + generate_pic_mask(
                line_points, IMAGE_SIZE=IMAGE_SIZE, pixel_value=i + 1, too_long=too_long
            )
            pic_direction_mask = pic_direction_mask + direction_mask

    # 限制像素值在0-3之间
    # import pdb; pdb.set_trace()
    pic_mask[visited] = 100  # 100是一个不可能的值，用于标记错误，也就是ignore value
    pic_tag[visited] = 0  # 利用pic_mask的值来限制pic_tag的值，0是背景，背景不计算loss
    pic_third_channel = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)  # 占位

    # 使用visited来过滤pic_direction_mask
    pic_direction_mask[visited] = 0

    # convert pic_direction_mask to 0-255
    pic_direction_mask = (pic_direction_mask + 1) * 127.5
    pic_direction_mask = pic_direction_mask.astype(np.uint8)
    # add the third channel
    direction_third_channel = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    # concatenate the third channel
    direction_png = np.stack(
        [
            pic_direction_mask[:, :, 0],
            pic_direction_mask[:, :, 1],
            direction_third_channel,
        ],
        axis=-1,
    )

    # concat pic_mask and pic_tag
    pic_three_channel = np.stack([pic_mask, pic_tag, pic_third_channel], axis=-1)
    # import pdb; pdb.set_trace()
    if VISUALIZATION:  # cannot visualize 3-channel image, meaningless
        # 可视化
        pic_mask = visualize_seg_map(pic_mask)
        # 保存可视化图片
        pic_mask.save(target_path_tag)
    else:
        if len_lines <= 255:  # 可以等于255，有255个instance的时候正好instance id为1-255
            # 保存图片
            # cv2.imwrite(target_path, pic_mask)
            cv2.imwrite(target_path_tag, pic_three_channel)
        else:
            # save the array directly
            target_path_tag = target_path_tag.replace(".png", ".npy")
            np.save(target_path_tag, pic_three_channel)

    # save direction png
    direction_png_path = target_path_dir.replace("-GT.png", "-GT-direction.png")
    cv2.imwrite(direction_png_path, direction_png)


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


import os
import json
import multiprocessing
from functools import partial


def read_json(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    return data


from multiprocessing import Value, Lock

# Initialize the counter and the lock
counter = Value("i", 0)  # 'i' stands for an integer type
lock = Lock()


def process_image(
    image_info, target_folder_direction, target_folder_mask_tag, data, IMAGE_SIZE
):
    image_path, image_name, relative_path = image_info
    target_subfolder_dir = os.path.join(target_folder_direction, relative_path)
    target_subfolder_tag = os.path.join(target_folder_mask_tag, relative_path)

    if not os.path.exists(target_subfolder_dir):
        os.makedirs(target_subfolder_dir)
    if not os.path.exists(target_subfolder_tag):
        os.makedirs(target_subfolder_tag)

    target_path_tag = os.path.join(target_subfolder_tag, image_name + "-GT.png")
    target_path_direction = os.path.join(target_subfolder_dir, image_name + "-GT.png")

    generate_gt(image_name, target_path_tag, target_path_direction, data, IMAGE_SIZE)
    # Increment the counter safely using the lock
    with lock:
        counter.value += 1
        if counter.value % 100 == 0:
            print(f"已处理{counter.value}张图片")


def collect_image_info(source_folder):
    image_info_list = []
    for root, _, files in os.walk(source_folder):
        for file in files:
            image_path = os.path.join(root, file)
            image_name = os.path.splitext(file)[0]
            relative_path = os.path.relpath(root, source_folder)
            image_info_list.append((image_path, image_name, relative_path))
    return image_info_list


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--file_name",
        type=str,
    )
    argparser.add_argument(
        "--source_folder",
        type=str,
    )
    argparser.add_argument(
        "--target_folder",
        type=str,
    )
    argparser.add_argument(
        "--image_size",
        type=int,
    )
    args = argparser.parse_args()

    IMAGE_SIZE = args.image_size

    file_name = args.file_name
    data = read_json(file_name)

    source_folder = args.source_folder
    target_folder_direction = args.target_folder + "_angle_direction"
    target_folder_mask_tag = args.target_folder + "-mask-tag"

    print("开始处理图片")
    image_info_list = collect_image_info(source_folder)

    pool = multiprocessing.Pool(processes=16)
    process_image_partial = partial(
        process_image,
        target_folder_direction=target_folder_direction,
        target_folder_mask_tag=target_folder_mask_tag,
        data=data,
        IMAGE_SIZE=IMAGE_SIZE,
    )

    for i, _ in enumerate(
        pool.imap_unordered(process_image_partial, image_info_list), 1
    ):
        if i % 100 == 0:
            print("已处理{}张图片".format(i))

    pool.close()
    pool.join()
