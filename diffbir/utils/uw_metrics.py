import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import sys
import os
from pathlib import Path

# 添加UW_Eval到路径
UW_EVAL_PATH = Path(__file__).parent / "UW_Eval"
if str(UW_EVAL_PATH) not in sys.path:
    sys.path.append(str(UW_EVAL_PATH))

from core.Losses import CustomMetric
from core.Losses.NoRef import calc_uciqe, calc_uiqm, calc_niqe


class UnderwaterMetrics(nn.Module):
    """水下图像质量指标计算器"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.custom_metric = CustomMetric()
        
    def cast_tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """将tensor转换为numpy数组，格式为HWC"""
        # 确保tensor在[0,1]范围内
        if tensor.max() <= 1.0:
            tensor = (tensor * 255).clamp(0, 255)
        
        # 转换为numpy并调整维度
        if tensor.dim() == 4:  # BCHW
            tensor = tensor.squeeze(0)  # 移除batch维度
        
        if tensor.shape[0] == 3:  # CHW
            tensor = tensor.permute(1, 2, 0)  # 转换为HWC
        
        return tensor.cpu().numpy().astype(np.uint8)
    
    def calculate_uiqm(self, image: torch.Tensor) -> Dict[str, float]:
        """计算UIQM指标"""
        image_np = self.cast_tensor_to_numpy(image)
        return calc_uiqm(image_np)
    
    def calculate_uciqe(self, image: torch.Tensor) -> Dict[str, float]:
        """计算UCIQE指标"""
        image_np = self.cast_tensor_to_numpy(image)
        return calc_uciqe(image_np)
    
    def calculate_niqe(self, image: torch.Tensor) -> Dict[str, float]:
        """计算NIQE指标"""
        image_np = self.cast_tensor_to_numpy(image)
        return calc_niqe(image_np)
    
    def calculate_all_metrics(self, pred: torch.Tensor, gt: torch.Tensor = None) -> Dict[str, float]:
        """计算所有水下图像质量指标"""
        metrics = {}
        
        # 计算预测图像的无参考指标
        uiqm_pred = self.calculate_uiqm(pred)
        uciqe_pred = self.calculate_uciqe(pred)
        niqe_pred = self.calculate_niqe(pred)
        
        metrics.update(uiqm_pred)
        metrics.update(uciqe_pred)
        metrics.update(niqe_pred)
        
        # 如果有GT，计算有参考指标
        if gt is not None:
            # 使用CustomMetric计算有参考指标
            ref_metrics = self.custom_metric(pred, gt)
            metrics.update(ref_metrics)
        
        return metrics
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor = None) -> Dict[str, float]:
        """前向传播，计算所有指标"""
        return self.calculate_all_metrics(pred, gt)


def calculate_underwater_metrics_batch(pred_batch: torch.Tensor, 
                                     gt_batch: torch.Tensor = None,
                                     device: str = "cuda") -> Dict[str, float]:
    """批量计算水下图像质量指标"""
    uw_metrics = UnderwaterMetrics(device=device)
    
    batch_size = pred_batch.shape[0]
    all_metrics = []
    
    for i in range(batch_size):
        pred = pred_batch[i:i+1]  # 保持batch维度
        gt = gt_batch[i:i+1] if gt_batch is not None else None
        
        metrics = uw_metrics(pred, gt)
        all_metrics.append(metrics)
    
    # 计算平均值
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    return avg_metrics 