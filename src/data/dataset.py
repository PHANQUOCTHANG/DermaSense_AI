"""
DermaSense AI — Dataset
Lớp Dataset tùy chỉnh cho cả bài toán Nhị phân (Stage 2) và Đa lớp (Stage 3).
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import yaml

class DermDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        is_binary: bool = False,
        transforms=None,
    ):
        """Khởi tạo Dataset.
        
        Args:
            img_dir: Đường dẫn đến thư mục chứa ảnh (vd: data/processed/train)
                     Bên trong có các thư mục con chứa ảnh theo tên bệnh.
            is_binary: Nếu True, sẽ map 7 lớp về 2 lớp (0: Low Risk, 1: High Risk).
                       Nếu False, giữ nguyên 7 lớp (0-6).
            transforms: Pipeline augmentation của albumentations.
        """
        self.img_dir = Path(img_dir)
        self.is_binary = is_binary
        self.transforms = transforms
        
        # Đọc cấu hình để lấy nhãn
        with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
            
        self.classes = base_config["labels"]["classes"]
        self.high_risk = base_config["labels"]["high_risk"]
        
        # Tạo từ điển map tên class sang index 0-6
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        self._load_samples()
        
    def _load_samples(self):
        """Quét thư mục để lấy danh sách toàn bộ ảnh và nhãn tương ứng."""
        for class_name in self.classes:
            class_dir = self.img_dir / class_name
            if not class_dir.exists():
                continue
                
            for img_path in class_dir.glob("*.jpg"):
                if self.is_binary:
                    # 1 nếu thuộc High Risk, 0 nếu Low Risk
                    label = 1 if class_name in self.high_risk else 0
                else:
                    label = self.class_to_idx[class_name]
                    
                self.samples.append((str(img_path), label))
                
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Đọc ảnh bằng OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Áp dụng Albumentations
        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image = augmented["image"]
            
        # Trả về FloatTensor cho features, tensor int64 cho label
        return image, torch.tensor(label, dtype=torch.long)
