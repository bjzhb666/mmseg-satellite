# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from mmseg.registry import METRICS


@METRICS.register_module()
class InstanceIoUMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
       
        
        num_classes = len(self.dataset_meta['classes'])
        num_line_type_classes = len(self.dataset_meta['line_type_classes'])
        num_line_num_classes = len(self.dataset_meta['line_num_classes'])

        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            # pred_direct_map_2048 = data_sample['pred_direct_map_2048']['data']
            # pred_tag_map_2048 = data_sample['pred_tag_map_2048']['data']
            # # add one extra dimension to use F.interpolate
            # pred_direct_map_2048 = pred_direct_map_2048.unsqueeze(1)
            # pred_tag_map_2048 = pred_tag_map_2048.unsqueeze(1)

            # # resize to 512*512
            # pred_direct_map_512 = F.interpolate(pred_direct_map_2048, size=(512, 512), mode='bilinear', align_corners=False)
            # pred_tag_map_512 = F.interpolate(pred_tag_map_2048, size=(512, 512), mode='bilinear',align_corners=False)
            
            # # remove the extra dimension
            # pred_direct_map_512 = pred_direct_map_512.squeeze(1)
            # pred_tag_map_512 = pred_tag_map_512.squeeze(1)
            
            pred_line_type_label = data_sample['pred_seg_line_type']['data'].squeeze()
            pred_line_num_label = data_sample['pred_seg_line_num']['data'].squeeze()
            # TODO: write clusting evalution code here
            DEBUG = True # set to True to save the prediction
            if DEBUG:
                img_name = osp.basename(data_sample['img_path'])[:-4]
                pred_label_cpu = pred_label.cpu().numpy()
                pred_line_type_label_cpu = pred_line_type_label.cpu().numpy()
                pred_line_num_label_cpu = pred_line_num_label.cpu().numpy()

                # pred_direct_map_512_cpu = pred_direct_map_512.cpu().numpy()
                # pred_tag_map_512_cpu = pred_tag_map_512.cpu().numpy()
                gt_seg = data_sample['gt_sem_seg']['data'].squeeze().cpu().numpy()
                gt_direction = data_sample['direction_map']['data'].squeeze().cpu().numpy()
                gt_instance_map = data_sample['gt_instance_map']['data'].squeeze().cpu().numpy()
                gt_line_type_map = data_sample['gt_line_type_map']['data'].squeeze().cpu().numpy()
                gt_line_num_map = data_sample['gt_line_num_map']['data'].squeeze().cpu().numpy()
                # save the prediction
                # save pred_label as png
                output_png = Image.fromarray(pred_label_cpu.astype(np.uint8))
                output_line_type_png = Image.fromarray(pred_line_type_label_cpu.astype(np.uint8))
                output_line_num_png = Image.fromarray(pred_line_num_label_cpu.astype(np.uint8))
                gt_seg_png = Image.fromarray(gt_seg.astype(np.uint8))
                gt_line_type_png = Image.fromarray(gt_line_type_map.astype(np.uint8))
                gt_line_num_png = Image.fromarray(gt_line_num_map.astype(np.uint8))
                gt_instance_png = Image.fromarray(gt_instance_map.astype(np.uint8))
                save_dir = './work_dirs/three_seg_head'
                mkdir_or_exist(f'{save_dir}')
                output_png.save(f'{save_dir}/pred_label-{img_name}.png')
                output_line_type_png.save(f'{save_dir}/pred_line_type-{img_name}.png')
                output_line_num_png.save(f'{save_dir}/pred_line_num-{img_name}.png')
                gt_seg_png.save(f'{save_dir}/gt_seg-{img_name}.png')
                gt_line_type_png.save(f'{save_dir}/gt_line_type-{img_name}.png')
                gt_line_num_png.save(f'{save_dir}/gt_line_num-{img_name}.png')
                gt_instance_png.save(f'{save_dir}/gt_instance-{img_name}.png')
                # np.save(f'{save_dir}/pred_direct_map_512-{img_name}.npy', pred_direct_map_512_cpu)
                # np.save(f'{save_dir}/pred_tag_map_512-{img_name}.npy', pred_tag_map_512_cpu)
                # np.save(f'{save_dir}/gt_direction-{img_name}.npy', gt_direction)
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)
                line_type_label = data_sample['gt_line_num_map']['data'].squeeze().to(
                    pred_line_type_label)
                line_num_label = data_sample['gt_line_num_map']['data'].squeeze().to(
                    pred_line_num_label)
                self.results.append(
                    self.intersect_and_union(pred_label, label, num_classes,
                                             self.ignore_index) + \
                    self.intersect_and_union(pred_line_type_label, line_type_label, num_line_type_classes,
                                             self.ignore_index) + \
                    self.intersect_and_union(pred_line_num_label, line_num_label, num_line_num_classes,
                                             self.ignore_index))
                
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                png_line_type_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}_line_type.png'))
                output_line_type_mask = pred_line_type_label.cpu().numpy()
                png_line_num_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}_line_num.png'))
                output_line_num_mask = pred_line_num_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                    output_line_type_mask = output_line_type_mask + 1
                    output_line_num_mask = output_line_num_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)
                output_line_type = Image.fromarray(output_line_type_mask.astype(np.uint8))
                output_line_type.save(png_line_type_filename)
                output_line_num = Image.fromarray(output_line_num_mask.astype(np.uint8))
                output_line_num.save(png_line_num_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 12

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])

        total_area_intersect_line_type = sum(results[4])
        total_area_union_line_type = sum(results[5])
        total_area_pred_line_type_label = sum(results[6])
        total_area_line_type_label = sum(results[7])

        total_area_intersect_line_num = sum(results[8])
        total_area_union_line_num = sum(results[9])
        total_area_pred_line_num_label = sum(results[10])
        total_area_line_num_label = sum(results[11])
                                         
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        class_names = self.dataset_meta['classes']

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
 
        ret_metrics_line_type = self.total_area_to_metrics(
            total_area_intersect_line_type, total_area_union_line_type, total_area_pred_line_type_label,
            total_area_line_type_label, self.metrics, self.nan_to_num, self.beta, metric_suffix='_line_type')
        
        line_type_class_name = self.dataset_meta['line_type_classes']
        
        ret_metrics_summary_line_type = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics_line_type.items()
        })

        metrics_line_type = dict()
        # import pdb; pdb.set_trace()
        for key, val in ret_metrics_summary_line_type.items():
            if key == 'aAcc_line_type':
                metrics_line_type[key] = val
            else:
                metrics_line_type['m' + key] = val
        
        # each class table
        ret_metrics_line_type.pop('aAcc_line_type', None)
        ret_metrics_class_line_type = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics_line_type.items()
        })
        ret_metrics_class_line_type.update({'Class': line_type_class_name})
        ret_metrics_class_line_type.move_to_end('Class', last=False)
        class_table_data_line_type = PrettyTable()
        for key, val in ret_metrics_class_line_type.items():
            class_table_data_line_type.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data_line_type.get_string(), logger=logger)
        
        ret_metrics_line_num = self.total_area_to_metrics(
            total_area_intersect_line_num, total_area_union_line_num, total_area_pred_line_num_label,
            total_area_line_num_label, self.metrics, self.nan_to_num, self.beta, metric_suffix='_line_num')
        
        line_num_class_name = self.dataset_meta['line_num_classes']
        
        ret_metrics_summary_line_num = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics_line_num.items()
        })

        metrics_line_num = dict()
        for key, val in ret_metrics_summary_line_num.items():
            if key == 'aAcc_line_num':
                metrics_line_num[key] = val
            else:
                metrics_line_num['m' + key] = val

        # each class table
        ret_metrics_line_num.pop('aAcc_line_num', None)
        ret_metrics_class_line_num = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics_line_num.items()
        })
        ret_metrics_class_line_num.update({'Class': line_num_class_name})
        ret_metrics_class_line_num.move_to_end('Class', last=False)
        class_table_data_line_num = PrettyTable()
        for key, val in ret_metrics_class_line_num.items():
            class_table_data_line_num.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data_line_num.get_string(), logger=logger)
        # import pdb; pdb.set_trace()
        metrics.update(metrics_line_type)
        metrics.update(metrics_line_num)
        return metrics

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1,
                              metric_suffix:str = ''):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
            metric_suffix (str): The suffix of the metric name. Default: ''.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric+metric_suffix: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics
