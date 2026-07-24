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
        use_clinical: bool = False,
        clinical_csv: Optional[str] = None
    ):
        """Khởi tạo Dataset.
        
        Args:
            img_dir: Đường dẫn đến thư mục chứa ảnh (vd: data/processed/train)
                     Bên trong có các thư mục con chứa ảnh theo tên bệnh.
            is_binary: Nếu True, sẽ map 7 lớp về 2 lớp (0: Low Risk, 1: High Risk).
                       Nếu False, giữ nguyên 7 lớp (0-6).
            transforms: Pipeline augmentation của albumentations.
            use_clinical: Cờ báo hiệu có sử dụng metadata lâm sàng hay không (Stage 3).
            clinical_csv: Đường dẫn đến file metadata_encoded.csv.
        """
        self.img_dir = Path(img_dir)
        self.is_binary = is_binary
        self.transforms = transforms
        self.use_clinical = use_clinical
        
        # Đọc metadata nếu dùng Multimodal
        self.metadata = None
        if self.use_clinical and clinical_csv is not None:
            df = pd.read_csv(clinical_csv)
            # Chuyển image_id thành index để tra cứu cho lẹ O(1)
            self.metadata = df.set_index('image_id')
        
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
        
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor, int]]:
        img_path_str, label = self.samples[idx]
        img_path = Path(img_path_str)
        
        # Đọc ảnh bằng OpenCV
        image = cv2.imread(img_path_str)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {img_path_str}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Áp dụng Albumentations
        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image = augmented["image"]
            
        label_tensor = torch.tensor(label, dtype=torch.long)
            
        # Trả về Multimodal nếu yêu cầu
        if self.use_clinical and self.metadata is not None:
            img_id = img_path.stem
            
            # Lấy dòng metadata tương ứng với image_id
            if img_id in self.metadata.index:
                row = self.metadata.loc[img_id]
                
                # Trích xuất 5 features (đã được encode thành số ở Stage 1)
                age = float(row.get('age', 0.0))
                sex = float(row.get('sex', 0.5))
                anatom_site = float(row.get('anatom_site', 0.0))
                duration = float(row.get('duration', 0.0))
                symptom = float(row.get('symptoms', 0.0))
                
                clinical_features = torch.tensor([age, sex, anatom_site, duration, symptom], dtype=torch.float32)
            else:
                # Nếu thiếu metadata, trả về vector 0
                clinical_features = torch.zeros(5, dtype=torch.float32)
                
            return image, clinical_features, label_tensor

        # Trả về FloatTensor cho features, tensor int64 cho label (Vision-only)
        return image, label_tensor
