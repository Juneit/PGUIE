from typing import Sequence, Dict, Union, List, Mapping, Any, Optional, Tuple
import math
import time
import io
import random

import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data

from .degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression,
)
from .utils import load_file_list, center_crop_arr, random_crop_arr
from ..utils.common import instantiate_from_config


class CodeformerDataset(data.Dataset):

    def __init__(
        self,
        file_list: str,
        lq_file_list: str,
        gt_file_list: str,
        file_backend_cfg: Mapping[str, Any],
        out_size: int,
        crop_type: str,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int],
        resize_type: str = "none",
    ) -> "CodeformerDataset":
        super(CodeformerDataset, self).__init__()
        self.lq_file_list = lq_file_list
        self.gt_file_list = gt_file_list
        self.lq_image_files = load_file_list(lq_file_list)
        self.gt_image_files = load_file_list(gt_file_list)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        self.resize_type = resize_type
        assert self.crop_type in ["none", "center", "random"]
        assert self.resize_type in ["none", "bilinear", "bicubic", "lanczos"]
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def aligned_random_crop_arr(self, pil_image, image_size) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        对齐的随机裁剪，返回裁剪后的图像和裁剪位置信息
        简化版本：只进行随机位置裁剪，不进行缩放操作
        """
        arr = np.array(pil_image)
        h, w = arr.shape[:2]
        
        # 确保图像足够大可以进行裁剪
        if h < image_size or w < image_size:
            raise ValueError(f"Image size ({h}, {w}) is smaller than target crop size {image_size}")
        
        # 随机选择裁剪位置
        crop_y = random.randrange(h - image_size + 1)
        crop_x = random.randrange(w - image_size + 1)
        
        # 执行裁剪
        cropped_arr = arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
        
        # 检查裁剪结果是否包含NaN或无穷大值
        if np.any(np.isnan(cropped_arr)) or np.any(np.isinf(cropped_arr)):
            raise ValueError(f"Cropped image contains NaN or Inf values")
        
        return cropped_arr, (crop_y, crop_x)

    def aligned_crop_with_position(self, pil_image, image_size, crop_position: Tuple[int, int]) -> np.ndarray:
        """
        根据给定的位置进行对齐裁剪
        简化版本：只进行位置裁剪，不进行缩放操作
        """
        arr = np.array(pil_image)
        h, w = arr.shape[:2]
        crop_y, crop_x = crop_position
        
        # 确保裁剪位置在有效范围内
        crop_y = min(crop_y, h - image_size)
        crop_x = min(crop_x, w - image_size)
        crop_y = max(0, crop_y)
        crop_x = max(0, crop_x)
        
        # 执行裁剪
        cropped_arr = arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
        
        # 检查裁剪结果是否包含NaN或无穷大值
        if np.any(np.isnan(cropped_arr)) or np.any(np.isinf(cropped_arr)):
            raise ValueError(f"Cropped image contains NaN or Inf values")
        
        return cropped_arr

    def load_gt_image(
        self, image_path: str, max_retry: int = 5, crop_position: Optional[Tuple[int, int]] = None
    ) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        if self.resize_type != "none":
            if image.height != self.out_size or image.width != self.out_size:
                if self.resize_type == "bilinear":
                    resample = Image.BILINEAR
                elif self.resize_type == "bicubic":
                    resample = Image.BICUBIC
                elif self.resize_type == "lanczos":
                    resample = Image.LANCZOS
                else:
                    resample = Image.BICUBIC
                
                image = image.resize((self.out_size, self.out_size), resample)
        
        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    if crop_position is not None:
                        # 使用对齐的裁剪位置
                        image = self.aligned_crop_with_position(image, self.out_size, crop_position)
                    else:
                        # 原有的随机裁剪方式
                        image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            image = np.array(image)
            if image.shape[0] != self.out_size or image.shape[1] != self.out_size:
                raise ValueError(f"Image size {image.shape[:2]} does not match out_size {self.out_size}")
        
        # hwc, rgb, 0,255, uint8
        return image

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        img_gt = None
        crop_position = None
        retry_count = 0
        max_retries = 10  # 防止无限递归
        
        while img_gt is None and retry_count < max_retries:
            # load meta file
            image_file = self.gt_image_files[index]
            gt_path = image_file["image_path"]
            prompt = image_file["prompt"]
            
            # 如果是random crop模式，先生成裁剪位置
            if self.crop_type == "random":
                # 先加载图像来确定裁剪位置
                image_bytes = None
                max_retry = 5
                while image_bytes is None:
                    if max_retry == 0:
                        break
                    image_bytes = self.file_backend.get(gt_path)
                    max_retry -= 1
                    if image_bytes is None:
                        time.sleep(0.5)
                
                if image_bytes is not None:
                    temp_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    # 检查图像大小是否足够进行裁剪
                    h, w = temp_image.size
                    if h < self.out_size or w < self.out_size:
                        print(f"Image size ({h}, {w}) is smaller than target crop size {self.out_size}, skipping {gt_path}")
                        index = random.randint(0, len(self) - 1)
                        retry_count += 1
                        continue
                    
                    try:
                        # 使用对齐的随机裁剪来获取位置
                        _, crop_position = self.aligned_random_crop_arr(temp_image, self.out_size)
                    except ValueError as e:
                        print(f"Failed to crop image {gt_path}: {e}, skipping")
                        index = random.randint(0, len(self) - 1)
                        retry_count += 1
                        continue
            
            img_gt = self.load_gt_image(gt_path, crop_position=crop_position)
            if img_gt is None:
                print(f"filed to load {gt_path}, try another image")
                index = random.randint(0, len(self) - 1)
                retry_count += 1

        # 如果重试次数过多，返回一个默认值或抛出异常
        if retry_count >= max_retries:
            raise RuntimeError(f"Failed to load valid image after {max_retries} retries")

        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        h, w, _ = img_gt.shape
        if np.random.uniform() < 0.5:
            prompt = ""

        # load lq image with the same crop position
        img_lq = None
        while img_lq is None:
            lq_path = self.lq_image_files[index]["image_path"]
            img_lq = self.load_gt_image(lq_path, crop_position=crop_position)
            if img_lq is None:
                print(f"filed to load {lq_path}, try again")
        img_lq = (img_lq[..., ::-1] / 255.0).astype(np.float32)

        
        # ------------------------ generate lq image ------------------------ #
        # blur
        # kernel = random_mixed_kernels(
        #     self.kernel_list,
        #     self.kernel_prob,
        #     self.blur_kernel_size,
        #     self.blur_sigma,
        #     self.blur_sigma,
        #     [-math.pi, math.pi],
        #     noise_range=None,
        # )
        # img_lq = cv2.filter2D(img_gt, -1, kernel)
        # # downsample
        # scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        # img_lq = cv2.resize(
        #     img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR
        # )
        # # noise
        # if self.noise_range is not None:
        #     img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # # jpeg compression
        # if self.jpeg_range is not None:
        #     img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)

        # # resize to original size
        # img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB, [-1, 1]
        gt = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        lq = img_lq[..., ::-1].astype(np.float32)
        # lq = (img_lq[..., ::-1] * 2 - 1).astype(np.float32)

        # 检查图像数据是否包含NaN或无穷大值
        if np.any(np.isnan(gt)) or np.any(np.isinf(gt)) or np.any(np.isnan(lq)) or np.any(np.isinf(lq)):
            print(f"Final output contains NaN or Inf values, skipping and trying next image")
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        # 检查数据范围是否合理
        if np.any(gt < -1.1) or np.any(gt > 1.1) or np.any(lq < -1.1) or np.any(lq > 1.1):
            print(f"Data range out of bounds - GT: [{gt.min():.3f}, {gt.max():.3f}], LQ: [{lq.min():.3f}, {lq.max():.3f}], skipping")
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        return gt, lq, prompt

    def __len__(self) -> int:
        return len(self.gt_image_files)
