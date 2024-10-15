import os
import random 
import cv2
import json
import copy
import numpy as np

color_dict = {
    '道路边缘':(0,255,0),
    '车道线': (0,0,255),
    '虚拟线': (255,255,0),
    '白':(0,0,255),
    '黄':(0,255,255),
    '其他':(0,0,0),
    '实线':(0,0,255),
    '粗实线':(26,30,158),
    '虚线':(255,0,0),
    '短粗虚线':(255,192,0),
    '待转区':(255,0,255),
    '导流区':(0,255,0),
    '引导线':(255,255,0),
    '停车位':(0,255,255),
    '是':(0,255,0),
    '否':(0,0,255),
    "双向":5,
    '无':2
}
output = 'output20point/visualdraw0.6/'
if not os.path.exists(output):
    os.makedirs(output)
# image_root = '/data_vdc/datasets/LaneLines/PC2_1408_yutu/'

image_root2 = 'tools/draw/'
js = json.load(open('20level_json/result.json','r'))
# js = json.load(open('flyover2_0_0_sat/result.json','r'))
# js = json.load()
c = 0
print(len(js))
# for name in js:
#     print(name)
#     c += 1
  
#     ori = cv2.imread(image_root2 + name)
    
#     img = copy.deepcopy(ori)
#     img2 = copy.deepcopy(ori)
#     img4 = copy.deepcopy(ori)
#     img5 = copy.deepcopy(ori)
#     img6 = copy.deepcopy(ori)

#     for line in js[name]['lines']:
      
#         img = cv2.polylines(img,[np.array(line['points'] ).astype(np.int32)], False, color_dict[line['category']], thickness=2)
#         color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
#         img2 = cv2.polylines(img2,[np.array(line['points'] ).astype(np.int32)], False, color, thickness=3)
#         img4 = cv2.polylines(img4,[np.array(line['points'] ).astype(np.int32)], False, color_dict[line['color']], thickness=3)
#         img5 = cv2.polylines(img5,[np.array(line['points'] ).astype(np.int32)], False, color_dict[line['line_type']], thickness=3)
#         img6 = cv2.polylines(img6,[np.array(line['points'] ).astype(np.int32)], False, color_dict[line['boundary']], thickness=color_dict[line['direction']])
 
#     # cv2.imwrite(output +name.replace('png','visual.png') ,np.vstack((np.hstack((ori,img2,img)),np.hstack((img4,img5,img6)))))
#     # 依次保存图片
#     cv2.imwrite(output +name.replace('.png','visual-ori.png') ,ori)
#     cv2.imwrite(output +name.replace('.png','visual-instance.png'), img2)
#     cv2.imwrite(output +name.replace('.png','visual-category.png'), img)
#     cv2.imwrite(output +name.replace('.png','visual-color.png'), img4)
#     cv2.imwrite(output +name.replace('.png','visual-line_type.png'), img5)
#     cv2.imwrite(output +name.replace('.png','visual-boundary.png'), img6)


import threading
import cv2
import copy
import random
alpha = 0.6
# 定义一个函数来处理每个图像
def process_image(name):
    print(name)
    c = 0
    ori = cv2.imread(image_root2 + name)
    
    img = copy.deepcopy(ori)
    img2 = copy.deepcopy(ori)
    img4 = copy.deepcopy(ori)
    img5 = copy.deepcopy(ori)
    img6 = copy.deepcopy(ori)

    # Create copies of the original images to draw the lines on
    overlay_img = img.copy()
    overlay_img2 = img2.copy()
    overlay_img4 = img4.copy()
    overlay_img5 = img5.copy()
    overlay_img6 = img6.copy()
    
    for line in js[name]['lines']:
        img = cv2.polylines(img,[np.array(line['points'] ).astype(np.int32)], False, color_dict[line['category']], thickness=2)
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        img2 = cv2.polylines(img2,[np.array(line['points'] ).astype(np.int32)], False, color, thickness=3)
        img4 = cv2.polylines(img4,[np.array(line['points'] ).astype(np.int32)], False, color_dict[line['color']], thickness=3)
        img5 = cv2.polylines(img5,[np.array(line['points'] ).astype(np.int32)], False, color_dict[line['line_type']], thickness=3)
        img6 = cv2.polylines(img6,[np.array(line['points'] ).astype(np.int32)], False, color_dict[line['boundary']], thickness=color_dict[line['direction']])
        # 在每个线条的端点绘制一个圆圈
        for point in line['points']:
            x, y = np.array(point).astype(np.int32)
            cv2.circle(img, (x, y), 9, color_dict[line['category']], -1)
            cv2.circle(img2, (x, y), 9, color, -1)
            cv2.circle(img4, (x, y), 9, color_dict[line['color']], -1)
            cv2.circle(img5, (x, y), 9, color_dict[line['line_type']], -1)
            cv2.circle(img6, (x, y), 9, color_dict[line['boundary']], -1)
    
    # Blend the images with the original ones
    cv2.addWeighted(overlay_img, alpha, img, 1 - alpha, 0, img)
    cv2.addWeighted(overlay_img2, alpha, img2, 1 - alpha, 0, img2)
    cv2.addWeighted(overlay_img4, alpha, img4, 1 - alpha, 0, img4)
    cv2.addWeighted(overlay_img5, alpha, img5, 1 - alpha, 0, img5)
    cv2.addWeighted(overlay_img6, alpha, img6, 1 - alpha, 0, img6)

    cv2.imwrite(output +name.replace('.png','visual-ori.png') ,ori)
    cv2.imwrite(output +name.replace('.png','visual-instance.png'), img2)
    cv2.imwrite(output +name.replace('.png','visual-category.png'), img)
    cv2.imwrite(output +name.replace('.png','visual-color.png'), img4)
    cv2.imwrite(output +name.replace('.png','visual-line_type.png'), img5)
    cv2.imwrite(output +name.replace('.png','visual-boundary.png'), img6)

# 创建一个线程池
threads = []

from concurrent.futures import ThreadPoolExecutor

# 设置线程池大小
num_threads = 32

# 创建线程池
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # 遍历 js 字典并提交任务到线程池
    for name in js:
        # print(name)
        # 如果name不存在，跳过
        if not os.path.exists(image_root2 + name):
            continue
        executor.submit(process_image, name)

# # 遍历 js 字典并创建线程
# for name in js:
#     t = threading.Thread(target=process_image, args=(name,))
#     threads.append(t)
#     t.start()

# # 等待所有线程完成
# for t in threads:
#     t.join()