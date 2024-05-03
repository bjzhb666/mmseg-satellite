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
from typing import List, Tuple
from torch import Tensor


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
    upsample network, used to upsample the feature map from (N, 1, 256, 256) to (N, 1, 2048, 2048)
    """
    def __init__(self):
        super(UpsampleNet, self).__init__()
        # 256 -> 512
        self.deconv1 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(1)
        # 512 -> 1024
        self.deconv2 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(1)
        # 1024 -> 2048
        self.deconv3 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(1)
        
    def forward(self, x):
        x = self.deconv1(x)
        x = self.bn1(x)        
        x = torch.relu(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.deconv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        return x


class GradualReduction(nn.Module):
    def __init__(self):
        super(GradualReduction, self).__init__()
        self.conv1 = nn.Conv2d(480, 128, kernel_size=1)  # 第一步降维到128
        self.relu = nn.ReLU()  # 激活函数
        self.conv2 = nn.Conv2d(128, 1, kernel_size=1)  # 第二步降维到1
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = SELayer(input_channels, reduction)
        # 修改处：将通道数改变的卷积层移动到SE层之后
        self.conv = nn.Conv2d(input_channels, 1, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = self.se(x)
        x = self.conv(x)
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

    def __init__(self, tag_type, ham_channels=512, SEreduction=16, ham_kwargs=dict(), **kwargs):
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

        if tag_type == 'direct':
            self.tag = nn.Conv2d(in_channels=480, out_channels=1, kernel_size=1, stride=1, padding=0)
        elif tag_type == 'gradual':
            self.tag = GradualReduction()
        elif tag_type == 'SE': # Squeeze-and-Excitation 
            # https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf
            self.tag = SEBlock(in_channels=480, reduction=SEreduction)
        else:
            raise ValueError('tag type not supported')

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
        """
        import pdb; pdb.set_trace()
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
        # seg head: apply a conv block to squeeze feature map
        x = self.squeeze(inputs)
        # apply hamburger module
        x = self.hamburger(x)

        # apply a conv block to align feature map
        output = self.align(x)
        output = self.cls_seg(output)
        return output, tag_map_2048

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
        seg_logits, tag_map_2048 = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
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
        seg_logits, tag_map_2048 = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)
