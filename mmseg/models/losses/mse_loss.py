import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS


@MODELS.register_module()
class MSERegressionLoss(nn.Module):
    """
    MSE loss for regression task.

    Args:
        loss_weight (float): Weight of loss.
    """
    def __init__(self, loss_weight=1.0):
        super(MSERegressionLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, front_position):
        '''
        Args:
            pred (torch.Tensor): The prediction with shape (N, C, H, W).
            target (torch.Tensor): The learning target of the prediction with shape (N, C, H, W).
            front_position (torch.Tensor): The mask of the front position with shape (N, 1, H, W). 
                True for front position and False for background position.
        Returns:
            torch.Tensor: The calculated loss.
        '''
        # 使用front_position作为mask提取对应位置的input和gt
        masked_input = torch.masked_select(pred, front_position)
        masked_gt = torch.masked_select(target, front_position)
        # 计算masked tensors的MSE loss
        loss = F.mse_loss(masked_input, masked_gt, reduction='mean')
        
        return loss*self.loss_weight
