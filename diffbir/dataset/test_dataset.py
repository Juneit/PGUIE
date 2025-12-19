from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from .codeformer import CodeformerDataset
import random

class TestDataset(CodeformerDataset):
    """测试数据集类，支持没有gt文件的情况"""
    
    def __init__(self, has_gt: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_gt = has_gt  # 是否包含gt文件
        
    def __getitem__(self, index: int) -> Union[Tuple[np.ndarray, np.ndarray, str], Tuple[np.ndarray, str]]:
        """获取数据项
        
        Args:
            index: 数据索引
            
        Returns:
            如果has_gt为True，返回(gt, lq, prompt)
            如果has_gt为False，返回(lq, prompt)
        """
        # 获取低质量图像
        img_lq = None
        while img_lq is None:
            lq_path = self.lq_image_files[index]["image_path"]
            img_lq = self.load_gt_image(lq_path)
            if img_lq is None:
                print(f"failed to load {lq_path}, try again")
        
        # 转换为标准格式
        img_lq = (img_lq[..., ::-1] / 255.0).astype(np.float32)
        lq = img_lq[..., ::-1].astype(np.float32)  # BGR to RGB, [0, 1]
        
        # 如果有gt文件，则获取gt
        if self.has_gt:
            # 获取高质量图像
            img_gt = None
            while img_gt is None:
                image_file = self.gt_image_files[index]
                gt_path = image_file["image_path"]
                prompt = image_file["prompt"]
                img_gt = self.load_gt_image(gt_path)
                if img_gt is None:
                    print(f"failed to load {gt_path}, try another image")
                    index = random.randint(0, len(self) - 1)
            
            # 转换为标准格式
            img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
            gt = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)  # BGR to RGB, [-1, 1]
            
            return gt, lq, prompt
        else:
            # 没有gt文件，只返回lq和空prompt
            return lq, "" 