from typing import List
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import os.path as osp
import mmengine.fileio as fileio
import mmengine
import json
import torch
import torchvision
from IPython import embed
import torch.distributed as dist
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.special import comb as n_over_k

@ DATASETS.register_module()
class SatelliteSegDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('background', 'lines'),
        palette=[[0, 0, 0], [255, 255, 255]]) # 分割调色板
    
    def __init__(self, img_suffix='.png', seg_map_suffix='-GT.png', **kwargs) -> None:

        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
        
@ DATASETS.register_module()
class SatelliteInstanceDataset(BaseSegDataset):

    METAINFO = dict(
        # classes=('background', 'solid lane_line', 'dashed lane_line','short dashed lane_line', 'thick solid lane_line','other lane_line',
        #          'parking lot lane_line', 'curb', 'virtual_line'),
        # palette=[[0, 0, 0], [255, 255, 255], [255,0,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255], [0,255,0], [128,128,128]], # 分割调色板
        # Four class (ignore solid dashed ect.)
        # classes=('background', 'lane_line', 'curb', 'virtual_line'),
        # palette=[[0, 0, 0], [255, 255, 255], [255,0,0], [0,0,255]], # 分割调色板
        # Potential classes for zoom 19, ignore 'thick solid lane_line','other lane_line',
        # classes=('background', 'solid lane_line', 'dashed lane_line','short dashed lane_line',
        #          'parking lot lane_line', 'curb', 'virtual_line'),
        # palette=[[0, 0, 0], [255, 255, 255], [255,0,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255]], # 分割调色板
        # move parking to others
        classes=('background', 'solid lane_line', 'dashed lane_line','short dashed lane_line', 'thick solid lane_line','other lane_line',
                 'curb', 'virtual_line'),
        palette=[[0, 0, 0], [255, 255, 255], [255,0,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255], [0,255,0]], # 分割调色板
        color_classes=('background', 'white', 'yellow', 'others', 'none'),
        color_palette=[[0, 0, 0], [255, 255, 255], [255, 255, 0], [0, 0, 255], [0, 255, 255]],
        line_type_classes=('background','导流区', '实线', '虚线', '停车位', '短粗虚线', '粗实线', '其他', '待转区', '引导线', '无'),
        line_type_palette=[[0, 0, 0], [255, 165, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                            [255, 0, 255], [0, 255, 255], [0, 0, 128], [255, 255, 255], [0, 128, 0]],
        line_num_classes=('background','单线', '双线', '其他', '无'),
        line_num_palette=[[0, 0, 0], [255, 165, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]],
        attribute_classes=('background','无', '禁停网格', '减速车道', '公交车道', '其他', '潮汐车道', '借道区', '可变车道'),
        attribute_palette=[[0, 0, 0], [255, 165, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], 
                 [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 128, 128]],
        whether_bidirection_classes=('background', '无', '双向'),
        whether_bidirection_palette=[[0, 0, 0],[255, 165, 0], [255, 0, 0]],
        whether_boundary_classes=('background', 'yes', 'no'),
        whether_boundary_palette=[[0, 0, 0],[255, 165, 0], [255, 0, 0]]
    )

    
    def __init__(self, direction_path, color_path, line_type_path, line_num_path, 
                 attribute_path, ifbidirection_path, ifboundary_path,
                 img_suffix='.png', seg_map_suffix='-GT.png',
                 direction_map_suffix='-GT.png',  **kwargs) -> None:

        self.direction_path = direction_path
        self.color_path = color_path
        self.line_type_path = line_type_path
        self.line_num_path = line_num_path
        self.attribute_path = attribute_path
        self.ifbidirection_path = ifbidirection_path
        self.ifboundary_path = ifboundary_path

        self.direction_map_suffix = direction_map_suffix
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
        
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        direction_dir = osp.join(self.data_root, self.direction_path)
        color_dir = osp.join(self.data_root, self.color_path)
        line_type_dir = osp.join(self.data_root, self.line_type_path)
        line_num_dir = osp.join(self.data_root, self.line_num_path)
        attribute_dir = osp.join(self.data_root, self.attribute_path)
        ifbidirection_dir = osp.join(self.data_root, self.ifbidirection_path)
        ifboundary_dir = osp.join(self.data_root, self.ifboundary_path)

        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else: # NOTE: not given the ann_file (ours)
            _suffix_len = len(self.img_suffix)
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))
                
                if ann_dir is not None:
                    seg_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                if direction_dir is not None:
                    direction_map = img[:-_suffix_len] + self.direction_map_suffix
                    data_info['direction_path'] = osp.join(direction_dir, direction_map)
                if color_dir is not None:
                    color_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['color_path'] = osp.join(color_dir, color_map)
                if line_type_dir is not None:
                    line_type_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['line_type_path'] = osp.join(line_type_dir, line_type_map)
                if line_num_dir is not None:
                    line_num_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['line_num_path'] = osp.join(line_num_dir, line_num_map)
                if attribute_dir is not None:
                    attribute_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['attribute_path'] = osp.join(attribute_dir, attribute_map)
                if ifbidirection_dir is not None:
                    ifbidirection_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['ifbidirection_path'] = osp.join(ifbidirection_dir, ifbidirection_map)
                if ifboundary_dir is not None:
                    ifboundary_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['ifboundary_path'] = osp.join(ifboundary_dir, ifboundary_map)

                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list

# @DATASETS.register_module()
# class SatelliteDataset(BaseSegDataset):

#     METAINFO = dict(
#         classes=('background', 'lines'),
#         palette=[[0, 0, 0], [255, 255, 255]], # 分割调色板
#         line_classes=('road'),
#         line_palette=[[255, 165, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 0, 0]]) # 线的调色板

#     def __init__(self, json_path, img_suffix='.png', seg_map_suffix='-GT.png', bezier_order=2, 
#                  cache=True, cache_path=None, **kwargs) -> None:
#         # import pdb; pdb.set_trace() # 通过 **kwargs 将父类的参数传递给父类的构造函数
#         self.json_path = json_path
#         self.bezier_order = bezier_order
#         self.cache = cache
#         self.cache_path = cache_path
#         super().__init__(
#             img_suffix=img_suffix,
#             seg_map_suffix=seg_map_suffix,
#             **kwargs)
        
        

#     def load_data_list(self) -> List[dict]:
#         """Load annotation from directory or annotation file.

#         Returns:
#             list[dict]: All data info of dataset.
#         """
#         if self.cache:
#             data_list = torch.load(self.cache_path)
#             return data_list
        
#         data_list = []
#         img_dir = self.data_prefix.get('img_path', None)
#         ann_dir = self.data_prefix.get('seg_map_path', None)

#         self.cocodetection = CocoDetection(img_folder=img_dir, ann_file=self.json_path, 
#                                            transforms=None, bezier_order=self.bezier_order)
#         cache_lines = self.cocodetection.cache_lines()

#         # get file_name to id mapping
#         file_name_to_id = {}
#         json_file = json.load(open(self.json_path, 'r'))
#         for i in range(len(json_file['images'])):
#             file_name_to_id[json_file['images'][i]['file_name']] = json_file['images'][i]['id']


#         if not osp.isdir(self.ann_file) and self.ann_file:
#             assert osp.isfile(self.ann_file), \
#                 f'Failed to load `ann_file` {self.ann_file}'
#             lines = mmengine.list_from_file(
#                 self.ann_file, backend_args=self.backend_args)
#             for line in lines:
#                 img_name = line.strip()
#                 data_info = dict(
#                     img_path=osp.join(img_dir, img_name + self.img_suffix))
#                 if ann_dir is not None:
#                     seg_map = img_name + self.seg_map_suffix
#                     data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
#                 data_info['label_map'] = self.label_map
#                 data_info['reduce_zero_label'] = self.reduce_zero_label
#                 data_info['seg_fields'] = []
#                 data_list.append(data_info)
#         else: # not given the ann_file (ours)
#             _suffix_len = len(self.img_suffix)
            
#             # json_file = json.load(open(self.json_path, 'r'))
            
#             for img in fileio.list_dir_or_file( 
#                     dir_path=img_dir,
#                     list_dir=False,
#                     suffix=self.img_suffix,
#                     recursive=True,
#                     backend_args=self.backend_args):
#                 # img is file name like 'Cities1to30_30cm_BBA_BGRN_22Q1_1303033003100_3_8.png'
#                 data_info = dict(img_path=osp.join(img_dir, img))
#                 if ann_dir is not None:
#                     seg_map = img[:-_suffix_len] + self.seg_map_suffix
#                     data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
#                 data_info['label_map'] = self.label_map
#                 data_info['reduce_zero_label'] = self.reduce_zero_label
#                 data_info['seg_fields'] = []

#                 # get lines GT
#                 img_id = file_name_to_id[img]
#                 data_info['lines'] = cache_lines[img_id]['lines']
#                 data_info['lines_labels'] = cache_lines[img_id]['labels']
#                 data_list.append(data_info)
#             data_list = sorted(data_list, key=lambda x: x['img_path'])
#         # import pdb; pdb.set_trace()
#         return data_list


# class CocoDetection(torchvision.datasets.CocoDetection):
#     def __init__(self, img_folder, ann_file, transforms, bezier_order=2):
#         super(CocoDetection, self).__init__(img_folder, ann_file)
#         self._transforms = transforms
#         # self.prepare = ConvertCocoPolysToMask() # for line segments
#         self.prepare = ConvertCocoPolysToMaskCurve(bezier_order=bezier_order) # for curve segments

#     def __getitem__(self, idx):
#         img, target = super(CocoDetection, self).__getitem__(idx)
#         image_id = self.ids[idx]
#         target = {'image_id': image_id, 'annotations': target}
#         img, target = self.prepare(img, target)
#         if self._transforms is not None:
#             img, target = self._transforms(img, target)
#         return img, target

#     def cache_lines(self):
#         cache = {}
#         for i in range(len(self)):
#             cache[self[i][1]['image_id'].item()] = self[i][1]
#             if i % 100 == 0:
#                 print(f'caching {i} images')
#         return cache

# class ConvertCocoPolysToMask(object):

#     def __call__(self, image, target):
#         w, h = image.size

#         image_id = target["image_id"]
#         image_id = torch.tensor([image_id])

#         anno = target["annotations"]

#         anno = [obj for obj in anno]
 
#         lines = [obj["line"] for obj in anno]
#         lines = torch.as_tensor(lines, dtype=torch.float32).reshape(-1, 4)

#         lines[:, 2:] += lines[:, :2] #xyxy, convert line_val/train2017.json form xydxdy to xyxy again

#         lines[:, 0::2].clamp_(min=0, max=w)
#         lines[:, 1::2].clamp_(min=0, max=h)

#         classes = [obj["category_id"] for obj in anno]
#         classes = torch.tensor(classes, dtype=torch.int64)

#         target = {}
#         target["lines"] = lines
        

#         target["labels"] = classes
        
#         target["image_id"] = image_id

#         # for conversion to coco api
#         area = torch.tensor([obj["area"] for obj in anno])
#         iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
#         target["area"] = area
#         target["iscrowd"] = iscrowd

#         target["orig_size"] = torch.as_tensor([int(h), int(w)])
#         target["size"] = torch.as_tensor([int(h), int(w)])

#         return image, target


# class ConvertCocoPolysToMaskCurve(object):
#     def __init__(self, bezier_order=2):
#         self.bezier_order = bezier_order

#     def __call__(self, image, target): 
#         w, h = image.size

#         image_id = target["image_id"]
#         image_id = torch.tensor([image_id])

#         anno = target["annotations"]

#         anno = [obj for obj in anno]

#         lines = [np.array(obj["line"]) for obj in anno]
#         # lines_tensor = [torch.as_tensor(line, dtype=torch.float32) for line in lines]

#         line_params = []
#         for i in lines: # instance level loop
#             i[0::2] = np.clip(i[0::2], a_min=0, a_max=w)
#             i[1::2] = np.clip(i[1::2], a_min=0, a_max=h)
#             i = i.reshape(-1, 2) # (n, 2)
#             # copy from pytorch-auto-drive/gen_bezier_annotations.py
#             if i.shape[0] == 0:
#                 continue
#             fcns = BezierCurve(order=self.bezier_order)
#             is_success = fcns.get_control_points(i[:, 0], i[:, 1], interpolate=False)
#             if not is_success:
#                 print('image_id:', image_id.item())
#                 continue
#             matrix = fcns.save_control_points()
#             flatten = [round(p, 3) for sub_m in matrix for p in sub_m]
#             # import pdb; pdb.set_trace()
#             line_params.append(flatten)

#         line_params = torch.tensor(np.array(line_params), dtype=torch.float32).reshape(-1, 2*(self.bezier_order+1)) 
#         # line_params is in xyxyxy form
#         classes = [obj["category_id"] for obj in anno]
#         classes = torch.tensor(classes, dtype=torch.int64)

#         target = {}
#         target["lines"] = line_params # list of tensors
        

#         target["labels"] = classes
        
#         target["image_id"] = image_id

#         # for conversion to coco api
#         area = torch.tensor([obj["area"] for obj in anno])
#         iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
#         target["area"] = area
#         target["iscrowd"] = iscrowd

#         target["orig_size"] = torch.as_tensor([int(h), int(w)])
#         target["size"] = torch.as_tensor([int(h), int(w)])

#         return image, target


# class BezierCurve(object):
#     # Define Bezier curves for curve fitting
#     def __init__(self, order, num_sample_points=150):
#         self.num_point = order + 1
#         self.control_points = []
#         self.bezier_coeff = self.get_bezier_coefficient()
#         self.num_sample_points = num_sample_points
#         self.c_matrix = self.get_bernstein_matrix()

#     def get_bezier_coefficient(self):
#         Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
#         BezierCoeff = lambda ts: [[Mtk(self.num_point - 1, t, k) for k in range(self.num_point)] for t in ts]

#         return BezierCoeff

#     def interpolate_lane(self, x, y, n=50):
#         # Spline interpolation of a lane. Used on the predictions
#         assert len(x) == len(y)

#         tck, _ = splprep([x, y], s=0, t=n, k=min(3, len(x) - 1))

#         u = np.linspace(0., 1., n)
#         return np.array(splev(u, tck)).T

#     def get_control_points(self, x, y, interpolate=False):
#         if interpolate:
#             points = self.interpolate_lane(x, y)
#             x = np.array([x for x, _ in points])
#             y = np.array([y for _, y in points])

#         middle_points = self.get_middle_control_points(x, y)
#         if middle_points is None:
#             return False
#         for idx in range(0, len(middle_points) - 1, 2):
#             self.control_points.append([middle_points[idx], middle_points[idx + 1]])
#         return True
    
#     def get_bernstein_matrix(self):
#         tokens = np.linspace(0, 1, self.num_sample_points)
#         c_matrix = self.bezier_coeff(tokens)
#         return np.array(c_matrix)

#     def save_control_points(self):
#         return self.control_points

#     def assign_control_points(self, control_points):
#         self.control_points = control_points

#     def quick_sample_point(self, image_size=None):
#         control_points_matrix = np.array(self.control_points)
#         sample_points = self.c_matrix.dot(control_points_matrix)
#         if image_size is not None:
#             sample_points[:, 0] = sample_points[:, 0] * image_size[-1]
#             sample_points[:, -1] = sample_points[:, -1] * image_size[0]
#         return sample_points

#     def get_sample_point(self, n=50, image_size=None):
#         '''
#             :param n: the number of sampled points
#             :return: a list of sampled points
#         '''
#         t = np.linspace(0, 1, n)
#         coeff_matrix = np.array(self.bezier_coeff(t))
#         control_points_matrix = np.array(self.control_points)
#         sample_points = coeff_matrix.dot(control_points_matrix)
#         if image_size is not None:
#             sample_points[:, 0] = sample_points[:, 0] * image_size[-1]
#             sample_points[:, -1] = sample_points[:, -1] * image_size[0]

#         return sample_points

#     def get_middle_control_points(self, x, y):
#         dy = y[1:] - y[:-1]
#         dx = x[1:] - x[:-1]
#         dt = (dx ** 2 + dy ** 2) ** 0.5
#         if dt.sum() == 0: # wrong line
#             import pdb; pdb.set_trace()
#             # print('dt.sum() == 0')
#             return None
#         t = dt / dt.sum()
#         t = np.hstack(([0], t))
#         t = t.cumsum()
#         data = np.column_stack((x, y))
#         Pseudoinverse = np.linalg.pinv(self.bezier_coeff(t))  # (9,4) -> (4,9)
#         control_points = Pseudoinverse.dot(data)  # (4,9)*(9,2) -> (4,2)
#         medi_ctp = control_points[:, :].flatten().tolist()

#         return medi_ctp

# if __name__ == '__main__':
#     img_folder = '/data1/zhaohongbo/satellite/mmsegmentation/data/satellite/img_dir/train'
#     ann_file = '/data1/zhaohongbo/satellite/mmsegmentation/data/satellite/anno_json/lines_train2017.json'
#     dataset = CocoDetection(img_folder, ann_file, transforms=None)
#     cache = dataset.cache_all_images()
#     embed()

