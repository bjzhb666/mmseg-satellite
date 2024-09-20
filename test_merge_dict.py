# def merge_dicts_in_tuple(tuple_of_dicts):
#     merged_dict = {
#         'coco_gt': {
#             'annotations': [],
#             'images': [],
#         },
#         'coco_dt': [],
#     }
    
#     for d in tuple_of_dicts:
#         merged_dict['coco_gt']['annotations'].extend(d['coco_gt']['annotations'])
#         merged_dict['coco_gt']['images'].extend(d['coco_gt']['images'])
#         merged_dict['coco_dt'].extend(d['coco_dt'])
    
#     return merged_dict

# # 示例元组
# tuple_of_dicts = (
#     {
#         'coco_gt': {
#             'annotations': [1, 2],
#             'images': ['image1', 'image2'],
#         },
#         'coco_dt': ['dt1', 'dt2'],
#     },
#     {
#         'coco_gt': {
#             'annotations': [3, 4],
#             'images': ['image3', 'image4'],
#         },
#         'coco_dt': ['dt3', 'dt4'],
#     }
# )

# # 合并后的字典
# merged_dict = merge_dicts_in_tuple(tuple_of_dicts)
# print(merged_dict)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 示例数据
coco_gt_dict = {
    "images": [
        {"id": 1, "file_name": "image1.jpg", "height": 480, "width": 640},
        {"id": 2, "file_name": "image2.jpg", "height": 480, "width": 640},  # 背景图片，没有前景
        {"id": 3, "file_name": "image3.jpg", "height": 480, "width": 640}
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[10, 10, 20, 10, 20, 20, 10, 20]], "area": 100, "bbox": [10, 10, 10, 10], "iscrowd": 0},
        {"id": 2, "image_id": 3, "category_id": 1, "segmentation": [[15, 15, 30, 15, 30, 30, 15, 30]], "area": 150, "bbox": [15, 15, 15, 15], "iscrowd": 0}
    ],
    "categories": [
        {"id": 1, "name": "category1"}
    ]
}

coco_dt_list = [
    {"image_id": 1, "category_id": 1, "segmentation": [[10, 10, 20, 10, 20, 20, 10, 20]], "area": 100, "bbox": [10, 10, 10, 10], "score": 0.9},
    {"image_id": 2, "category_id": 1, "segmentation": [[12, 12, 22, 12, 22, 22, 12, 22]], "area": 100, "bbox": [12, 12, 10, 10], "score": 0.85},  # image2.jpg 的预测前景
    {"image_id": 3, "category_id": 1, "segmentation": [[15, 15, 30, 15, 30, 30, 15, 30]], "area": 150, "bbox": [15, 15, 15, 15], "score": 0.8}
]

# 将字典加载为 COCO 格式
coco_gt = COCO()
coco_gt.dataset = coco_gt_dict
coco_gt.createIndex()

# 加载检测结果
coco_dt = coco_gt.loadRes(coco_dt_list)

# 评估结果
coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

