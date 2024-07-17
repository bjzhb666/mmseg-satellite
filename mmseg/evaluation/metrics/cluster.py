import cv2
import numpy as np
from .eval_utils import (get_position, get_instance_image, sample_from_positions, 
                         generate_line_mask_without_direction, debug_instance_pred, prepare_coco_dict)
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.nn as nn

def watershed(binary_image, position):
    binary_image = (binary_image * 255).astype(np.uint8)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=3)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=5)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR), markers)
    y_pred = np.array([markers[pos[1], pos[0]] for pos in position])

    return y_pred - 1

def watercluster(label, pred_label, seg_probs, gt_instance, data_sample, 
                 GT_without_Water, save_instance_pred, use_seg_GT, minimal_area=1):
    per_class_gt_label = [(label == index).bool() for index in [1, 2, 3]] # list of [H, W], len=3
    per_class_dt_label = [(pred_label == index).bool() for index in [1, 2, 3]] # list of [H, W]


    gt_positions = [get_position(gt_mask.cpu().numpy()) for gt_mask in per_class_gt_label] # list of [N, 2], len=3, N is the number of points in the mask class i
    dt_positions = [get_position(dt_mask.cpu().numpy()) for dt_mask in per_class_dt_label] # [N, 2]
    dt_instance = np.zeros_like(gt_instance.cpu().numpy()) # [H, W]
    dt_instance_ori = np.zeros_like(gt_instance.cpu().numpy()) # [H, W], only for drawing
    
    for index, (dt_mask, dt_position, gt_mask, gt_position) in enumerate(zip(per_class_dt_label, dt_positions, per_class_gt_label, gt_positions)):
        # dt_position, gt_position are the positions of the points in class i
        # get instance predictions via watershed
        y_pred = watershed(dt_mask.cpu().numpy(), dt_position)
        y_pred[y_pred < 0] = 0

        # for dt_instance_ori, only for drawing
        instance_id_start = np.max(dt_instance)
        ori = get_instance_image(dt_position, dt_instance_ori.shape[0], dt_instance_ori.shape[1], y_pred)
        ori = ori + (ori != 0) * instance_id_start
        dt_instance_ori = dt_instance_ori + ori
        
        if GT_without_Water:
            y_pred = np.array(gt_instance[gt_position[:, 1], gt_position[:, 0]].cpu().numpy(), dtype=np.int32)
            dt_position = gt_position
            dt_mask = gt_mask
        # sample and reconstruct
        for instance_id in np.unique(y_pred):
            if instance_id == 0: continue

            instance_position = dt_position[y_pred == instance_id]
            # remove small regions
            if len(instance_position) <= minimal_area: continue # 面积小于100的实例不要

            sampled_positions = sample_from_positions(instance_position)
            rec = generate_line_mask_without_direction(
                coordinates=sampled_positions,
                line_width=1,
                image_size=dt_mask.cpu().numpy().shape,
            )
            if data_sample['ori_shape'][0] == 1024:
                rec = cv2.dilate(rec, kernel=np.ones((5, 5), dtype=np.uint8), iterations=1)
            elif data_sample['ori_shape'][0] == 512:
                rec = cv2.dilate(rec, kernel=np.ones((3, 3), dtype=np.uint8), iterations=1)

            # removing overlap
            if np.sum((rec == 255) * dt_instance) > 0: continue
            dt_instance = dt_instance + (rec == 255) * (instance_id + instance_id_start)

    item = {
        "seg_probs": seg_probs.cpu().numpy(),
        "gt_label": label.cpu().numpy(),
        "dt_label": pred_label.cpu().numpy(),
        "gt_instance": gt_instance.cpu().numpy(),
        "dt_instance": dt_instance,
        # "dt_instance": gt_instance.cpu().numpy(),
    }

    if save_instance_pred:
        debug_instance_pred(data_sample, item)

    coco_dict = prepare_coco_dict(data_sample, item, use_seg_GT)

    return coco_dict


def onehot_encoding(logits, dim=0):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def DBSCAN_cluster(segmentation, line_type_seg, direction, post_processor):
    '''
    :param seg_prob: torch.tensor, shape=[C, H, W], C is the number of classes, segmentation results for category
    :param line_type_seg: torch.tensor, shape=[C1, H, W], segmentation results for line type
    :param direction: torch.tensor, shape=[H, W], direction map
    '''
    segmentation = segmentation.softmax(0)
    seg_oh_prod = onehot_encoding(segmentation).cpu().numpy() # [C, H, W]
    line_type_prod = F.softmax(line_type_seg, dim=0)
    line_type_oh_prod = onehot_encoding(line_type_prod).cpu().numpy() # [C1, H, W]
    max_pool_1 = nn.MaxPool2d((1, 5), padding=(0, 2), stride=1)
    avg_pool_1 = nn.AvgPool2d((9, 5), padding=(4, 2), stride=1)
    max_pool_2 = nn.MaxPool2d((5, 1), padding=(2, 0), stride=1)
    avg_pool_2 = nn.AvgPool2d((5, 9), padding=(2, 4), stride=1)
    embedding = np.concatenate([seg_oh_prod, line_type_oh_prod, direction.cpu().numpy()], axis=0) # [C + C1 + 1, H, W]
    
    for i in range(1, seg_oh_prod.shape[0]): # 每个类别分别处理
        single_mask = seg_oh_prod[i].astype('uint8') # [H, W]
        single_embedding = embedding.permute(1, 2, 0) # [N, C + C1 + 1]

        single_class_inst_mask, single_class_inst_coords = post_processor.postprocess(single_mask, single_embedding)
        if single_class_inst_mask is None:
            continue

        num_inst = len(single_class_inst_coords)
        prob = segmentation[i]
        prob[single_class_inst_mask == 0] = 0
        nms_mask_1 = ((max_pool_1(prob.unsqueeze(0))[0] - prob) < 0.0001).cpu().numpy()
        avg_mask_1 = avg_pool_1(prob.unsqueeze(0))[0].cpu().numpy()
        nms_mask_2 = ((max_pool_2(prob.unsqueeze(0))[0] - prob) < 0.0001).cpu().numpy()
        avg_mask_2 = avg_pool_2(prob.unsqueeze(0))[0].cpu().numpy()
        vertical_mask = avg_mask_1 > avg_mask_2
        horizontal_mask = ~vertical_mask
        nms_mask = (vertical_mask & nms_mask_1) | (horizontal_mask & nms_mask_2)

def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image: binary segmentation result, shape=[H, W]
    :param kernel_size: int, kernel size for morphological operation
    :return: binary segmentation result after morphological operation, shape=[H, W]
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))
    #  [[0 0 1 0 0]
    #  [1 1 1 1 1]
    #  [1 1 1 1 1]
    #  [1 1 1 1 1]
    #  [0 0 1 0 0]]
    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image: binary segmentation result, shape=[H, W]
    :return: connected components result, a tuple contains (labels, stats)
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self, dbscan_eps=0.35, postprocess_min_samples=200):
        """

        """
        self.dbscan_eps = dbscan_eps
        self.postprocess_min_samples = postprocess_min_samples

    def _embedding_feats_dbscan_cluster(self, embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        from sklearn.cluster import MeanShift

        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.postprocess_min_samples)
        # db = MeanShift()
        try:
            features = StandardScaler().fit_transform(embedding_image_feats)
            db.fit(features)
        except Exception as err:
            # print(err)
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)

        num_clusters = len(unique_labels)
        # cluster_centers = db.components_

        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            # 'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 255)
        lane_embedding_feats = instance_seg_ret[idx]
        # idx_scale = np.vstack((idx[0] / 256.0, idx[1] / 512.0)).transpose()
        # lane_embedding_feats = np.hstack((lane_embedding_feats, idx_scale))
        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        mask: np.array, shape=[H, W], instance segmentation result, each pixel belongs to a instance
        lane_coords: list of np.array, shape=[N, 2], N is the number of lane, 2 is the x, y coordinate
        """
        # get embedding feats and coords
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )

        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
        )

        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1]], dtype=np.int)
        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']
        coord = get_lane_embedding_feats_result['lane_coordinates']

        if db_labels is None:
            return None, None

        lane_coords = []

        for index, label in enumerate(unique_labels.tolist()):
            if label == -1:
                continue
            idx = np.where(db_labels == label)
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))
            mask[pix_coord_idx] = label + 1
            lane_coords.append(coord[idx])

        return mask, lane_coords 
    

class LaneNetPostProcessor(object):
    '''
    lanenet post process for lane generation
    '''
    def __init__(self, dbscan_eps=0.35, postprocess_min_samples=200) -> None:
        self._cluster = _LaneNetCluster(dbscan_eps=dbscan_eps, postprocess_min_samples=postprocess_min_samples)
    
    def postprocess(self, binary_seg_result, instance_seg_result, mim_area_thresthold=100):
        '''
        :param binary_seg_result: binary segmentation result
        :param instance_seg_result: features for instance segmentation
        :param mim_area_thresthold: minimum area threshold
        '''
        # convert binary_seg_result
        binary_seg_result = np.array(binary_seg_result * 255, dytpe=np.uint8)

        # apply image morphological operation to fill the holes and reduce the small area
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)

        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

        labels = connect_components_analysis_ret[1] # [H, W]
        stats = connect_components_analysis_ret[2]
        for index, stat in enumerate(stats):
            if stat[4] < mim_area_thresthold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        # apply cluster
        mask, lane_coords = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result
        )

        # TODO: merge the results of label and mask


        return mask, lane_coords