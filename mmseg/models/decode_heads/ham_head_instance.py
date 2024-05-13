# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.device import get_device

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from mmseg.utils import ConfigType, SampleList
from typing import List, Tuple, Any
from torch import Tensor
import math


class Matrix_Decomposition_2D_Base(nn.Module):
    """Base class of 2D Matrix Decomposition.

    Args:
        MD_S (int): The number of spatial coefficient in
            Matrix Decomposition, it may be used for calculation
            of the number of latent dimension D in Matrix
            Decomposition. Defaults: 1.
        MD_R (int): The number of latent dimension R in
            Matrix Decomposition. Defaults: 64.
        train_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in training. Defaults: 6.
        eval_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in evaluation. Defaults: 7.
        inv_t (int): Inverted multiple number to make coefficient
            smaller in softmax. Defaults: 100.
        rand_init (bool): Whether to initialize randomly.
            Defaults: True.
    """

    def __init__(self,
                 MD_S=1,
                 MD_R=64,
                 train_steps=6,
                 eval_steps=7,
                 inv_t=100,
                 rand_init=True):
        super().__init__()

        self.S = MD_S
        self.R = MD_R

        self.train_steps = train_steps
        self.eval_steps = eval_steps

        self.inv_t = inv_t

        self.rand_init = rand_init

    def _build_bases(self, B, S, D, R, device=None):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        """Forward Function."""
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        D = C // self.S
        N = H * W
        x = x.view(B * self.S, D, N)
        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, device=x.device)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, device=x.device)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        x = x.view(B, C, H, W)

        return x


class NMF2D(Matrix_Decomposition_2D_Base):
    """Non-negative Matrix Factorization (NMF) module.

    It is inherited from ``Matrix_Decomposition_2D_Base`` module.
    """

    def __init__(self, args=dict()):
        super().__init__(**args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, device=None):
        """Build bases in initialization."""
        if device is None:
            device = get_device()
        bases = torch.rand((B * S, D, R)).to(device)
        bases = F.normalize(bases, dim=1)

        return bases

    def local_step(self, x, bases, coef):
        """Local step in iteration to renew bases and coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        """Compute coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    """Hamburger Module. It consists of one slice of "ham" (matrix
    decomposition) and two slices of "bread" (linear transformation).

    Args:
        ham_channels (int): Input and output channels of feature.
        ham_kwargs (dict): Config of matrix decomposition module.
        norm_cfg (dict | None): Config of norm layers.
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.ham_in = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None)

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham


class UpsampleNet(nn.Module):
    """
    upsample network, used to upsample the feature map from (N, C, 256, 256) to (N, C, 2048, 2048)
    """
    def __init__(self, channels=1):
        super(UpsampleNet, self).__init__()
        self.channels = channels
        
        # 256 -> 512
        self.deconv1 = nn.ConvTranspose2d(self.channels, self.channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(self.channels)
        # 512 -> 1024
        self.deconv2 = nn.ConvTranspose2d(self.channels, self.channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(self.channels)
        # 1024 -> 2048
        self.deconv3 = nn.ConvTranspose2d(self.channels, self.channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(self.channels)
        # Additional layers for upsampling because 256 -> 2048 needs more than 3 doublings
        # 2048 -> 2048, here we don't change the resolution but might improve the features
        self.deconv4 = nn.ConvTranspose2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(self.channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        
        # Final additional layer
        x = F.relu(self.bn4(self.deconv4(x)))
        
        return x

@MODELS.register_module()
class LightHamInstanceHead(BaseDecodeHead):
    """SegNeXt decode head.

    This decode head is the implementation of `SegNeXt: Rethinking
    Convolutional Attention Design for Semantic
    Segmentation <https://arxiv.org/abs/2209.08575>`_.
    Inspiration from https://github.com/visual-attention-network/segnext.

    Specifically, LightHamHead is inspired by HamNet from
    `Is Attention Better Than Matrix Decomposition?
    <https://arxiv.org/abs/2109.04553>`.

    Args:
        ham_channels (int): input channels for Hamburger.
            Defaults: 512.
        ham_kwargs (int): kwagrs for Ham. Defaults: dict().
    """

    def __init__(self, tag_type:dict[str, Any], direction_type:dict[str, Any], ham_channels=512, loss_instance_decode=dict(type='AELoss',
                     loss_weight=1.0, push_loss_factor=0.1, minimum_instance_pixels=0),
                     loss_direction_decode=dict(type='MSERegressionLoss', loss_weight=1.0),
                     ham_kwargs=dict(), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.upsample = UpsampleNet()
        self.direct_upsample = UpsampleNet()

        # tag_type initialization
        if tag_type['type'] in ['DirectReduction', 'SEBlock', 'GradualReduction']:
            self.tag = MODELS.build(tag_type)
        else:
            raise ValueError(f"tag type: {tag_type['type']} not supported")
        
        # direction head initialization
        if direction_type['type'] in ['DirectReduction', 'SEBlock', 'GradualReduction']:
            self.direction_head = MODELS.build(direction_type)
        else:
            raise ValueError(f"direction type: {direction_type['type']} not supported")
        
        # loss instance decode (AE loss)
        if isinstance(loss_instance_decode, dict):
            self.loss_instance_decode = MODELS.build(loss_instance_decode)
        elif isinstance(loss_instance_decode, (list, tuple)):
            self.loss_instance_decode = nn.ModuleList()
            for loss_instance in loss_instance_decode:
                self.loss_instance_decode.append(MODELS.build(loss_instance))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_instance_decode)}')
        
        # loss direction decode (regression loss)
        if isinstance(loss_direction_decode, dict):
            self.loss_direct_decode = MODELS.build(loss_direction_decode)
        elif isinstance(loss_direction_decode, (list, tuple)):
            self.loss_direct_decode = nn.ModuleList()
            for loss_direct in loss_direction_decode:
                self.loss_direct_decode.append(MODELS.build(loss_direct))
        else:
            raise TypeError(f'loss_direct_decode must be a dict or sequence of dict,\
                but got {type(loss_direction_decode)}')
        
    def forward(self, inputs):
        """Forward function.
        inputs: list of feature maps from different levels (list of Tensor):
            - inputs[0]: the feature map from the highest level torch.Size([bs, 32, 512, 512])
            - inputs[1]: the feature map from the second highest level torch.Size([bs, 64, 256, 256])
            - inputs[2]: the feature map from the third highest level torch.Size([bs, 160, 128, 128])
            - inputs[3]: the feature map from the fourth highest level torch.Size([bs, 256, 64, 64])
        Returns:
            output: the output feature map torch.Size([bs, num_cls, 256, 256])
            tag_map_2048: the output tag map torch.Size([bs, 1, 2048, 2048])
            direction_map_2048: the output direction map torch.Size([bs, 1, 2048, 2048])
        """

        inputs = self._transform_inputs(inputs) # choose the last three layers

        inputs = [
            resize(
                level,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for level in inputs
        ] # resize all feature maps to the same size: 256*256

        inputs = torch.cat(inputs, dim=1) # bs, 480, 256, 256
        
        # tag head: apply a conv block to squeeze feature map
        # First step: resize the feature map to bs,1,256,256 (get tag map)
        tag_map_256 = self.tag(inputs)
        # Second step: upsample the tag map to bs,1,2048,2048
        tag_map_2048 = self.upsample(tag_map_256)
        tag_map_2048 = torch.tanh(tag_map_2048)
        
        # seg head: apply a conv block to squeeze feature map
        x = self.squeeze(inputs)
        # apply hamburger module
        x = self.hamburger(x)

        # direction head: predict a direction for each pixel
        # First step: resize the feature map to bs,2,256,256 (get direction map)
        direction_map = self.direction_head(inputs)
        # Second step: upsample the direction map to bs,2,2048,2048
        direction_map_2048 = self.direct_upsample(direction_map)
        # turn to -pi to pi
        direction_map_2048 = torch.tanh(direction_map_2048) * math.pi

        # apply a conv block to align feature map
        output = self.align(x)
        output = self.cls_seg(output)
        return output, tag_map_2048, direction_map_2048
    
    def _get_ignore_map(self, batch_data_samples: SampleList, ignore_index: int) -> Tensor:

        ignore_map = [
            data_sample.gt_sem_seg.data==ignore_index for data_sample in batch_data_samples
        ]
        return torch.stack(ignore_map, dim=0)
    
    def _stack_batch_instance_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_instance_segs = [
            data_sample.gt_instance_map.data for data_sample in batch_data_samples
        ] # list: len=batch_size, each element is a tensor of shape (1, 2048, 2048)
        
        return torch.stack(gt_instance_segs, dim=0)
    
    def loss_instance_by_feat(self, tag_map_2048: Tensor, batch_data_samples: SampleList) -> dict:
        """Compute AE loss for instance segmentation.

        Args:
            tag_map_2048 (Tensor): The output tag map. Shape: (batch_size, 1, 2048, 2048)
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg` and `gt_instance_map`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        gt_instance = self._stack_batch_instance_gt(batch_data_samples) # shape: (batch_size, 1, 2048, 2048)
        ignore_map = self._get_ignore_map(batch_data_samples, self.ignore_index) # shape: (batch_size, 1, 2048, 2048)
        loss_instance = dict()
        
        if not isinstance(self.loss_instance_decode, nn.ModuleList):
            losses_instance_decode = [self.loss_instance_decode]
        else:
            losses_instance_decode = self.loss_instance_decode
        for loss_instance_decode in losses_instance_decode:
            if loss_instance_decode.loss_name not in loss_instance:
                loss_instance[loss_instance_decode.loss_name] = loss_instance_decode(
                    tag_map_2048, gt_instance, ignore_position = ignore_map)
            else:
                loss_instance[loss_instance_decode.loss_name] += loss_instance_decode(
                    tag_map_2048, gt_instance, ignore_position = ignore_map)
                
        return loss_instance

    def _stack_batch_direct_gt(self, batch_data_samples: SampleList) -> Tensor:
        
        gt_directions = [
            data_sample.direction_map.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_directions, dim=0)

    def loss_direct_by_feat(self, direct_map_2048: Tensor, batch_data_samples: SampleList) -> dict:
        """Compute regression loss for direction prediction.

        Args:
            direct_map_2048 (Tensor): The output direction map. Shape: (batch_size, 1, 2048, 2048)
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg` and `gt_instance_map`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        gt_direction = self._stack_batch_direct_gt(batch_data_samples)
        seg_label = self._stack_batch_gt(batch_data_samples)
        front_map = (seg_label != 0) & (seg_label != self.ignore_index)
        
        loss_direct = dict()
        if not isinstance(self.loss_direct_decode, nn.ModuleList):
            losses_direct_decode = [self.loss_direct_decode]
        else:
            losses_direct_decode = self.loss_direct_decode

        for loss_direct_decode in losses_direct_decode:
            if loss_direct_decode.loss_name not in loss_direct:
                loss_direct[loss_direct_decode.loss_name] = loss_direct_decode(
                    direct_map_2048, gt_direction, front_map)
            else:
                loss_direct[loss_direct_decode.loss_name] += loss_direct_decode(
                    direct_map_2048, gt_direction, front_map)
                
        return loss_direct

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, tag_map_2048, direct_map_2048 = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        # import pdb; pdb.set_trace()
        losses_instance = self.loss_instance_by_feat(tag_map_2048, batch_data_samples)
        # update losses dict with instance loss
        losses_direct = self.loss_direct_by_feat(direct_map_2048, batch_data_samples)
        losses.update(losses_instance)
        losses.update(losses_direct)

        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits, tag_map_2048, direct_map_2048 = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas), tag_map_2048, direct_map_2048
