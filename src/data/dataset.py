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

# ============================================================
# Bảng ánh xạ: Tên bệnh gốc DermNet → Tên bệnh trong config
# ============================================================
DERMNET_NAME_MAPPING = {
    "Acne and Rosacea Photos": "Acne and Rosacea",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": "Actinic Keratosis and Skin Cancer",
    "Atopic Dermatitis Photos": "Atopic Dermatitis",
    "Bullous Disease Photos": "Bullous Disease",
    "Cellulitis Impetigo and other Bacterial Infections": "Cellulitis and Bacterial Infections",
    "Eczema Photos": "Eczema",
    "Exanthems and Drug Eruptions": "Exanthems and Drug Eruptions",
    "Hair Loss Photos Alopecia and other Hair Diseases": "Hair Loss",
    "Herpes HPV and other STDs Photos": "Herpes and HPV",
    "Light Diseases and Disorders of Pigmentation": "Light Diseases and Pigmentation",
    "Lupus and other Connective Tissue diseases": "Lupus and Connective Tissue",
    "Melanoma Skin Cancer Nevi and Moles": "Melanoma and Nevi",
    "Nail Fungus and other Nail Disease": "Nail Fungus",
    "Poison Ivy Photos and other Contact Dermatitis": "Poison Ivy and Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases": "Psoriasis and Lichen Planus",
    "Scabies Lyme Disease and other Infestations and Bites": "Scabies and Lyme",
    "Seborrheic Keratoses and other Benign Tumors": "Seborrheic Keratoses and Benign Tumors",
    "Systemic Disease": "Systemic Diseases",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "Tinea and Fungal Infections",
    "Urticaria Hives": "Urticaria Hives",
    "Vascular Tumors": "Vascular Tumors",
    "Vasculitis Photos": "Vasculitis",
    "Warts Molluscum and other Viral Infections": "Warts and Molluscum",
}

# Bảng encode các biến lâm sàng từ text → số
SEX_MAP = {"male": 1.0, "female": 0.0}
SITE_MAP = {"head/neck": 0.0, "anterior torso": 0.2, "posterior torso": 0.4, "upper extremity": 0.6, "lower extremity": 0.8, "palms/soles": 1.0}
SYMPTOM_MAP = {"none": 0.0, "itch": 0.3, "pain": 0.6, "bleeding": 1.0}
SKIN_TYPE_MAP = {"I": 0.0, "II": 0.2, "III": 0.4, "IV": 0.6, "V": 0.8, "VI": 1.0}
FAMILY_HISTORY_MAP = {"yes": 1.0, "no": 0.0}

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
            is_binary: Nếu True, sẽ map 23 lớp về 2 lớp (0: Low Risk, 1: High Risk).
                       Nếu False, giữ nguyên 23 lớp (0-22).
            transforms: Pipeline augmentation của albumentations.
            use_clinical: Cờ báo hiệu có sử dụng metadata lâm sàng hay không (Stage 3).
            clinical_csv: Đường dẫn đến file metadata.csv hoặc metadata_encoded.csv.
        """
        self.img_dir = Path(img_dir)
        self.is_binary = is_binary
        self.transforms = transforms
        self.use_clinical = use_clinical
        
        # Đọc metadata nếu dùng Multimodal
        self.metadata = None
        if self.use_clinical and clinical_csv is not None:
            if Path(clinical_csv).exists():
                df = pd.read_csv(clinical_csv)
                # Chuyển image_id thành index để tra cứu cho lẹ O(1)
                self.metadata = df.set_index('image_id')
        
        # Đọc cấu hình để lấy nhãn
        with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
            
        self.classes = base_config["labels"]["classes"]
        self.high_risk = base_config["labels"]["high_risk"]
        
        # Tạo từ điển map tên class sang index 0-22
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        self._load_samples()
        
    def _normalize_class_name(self, raw_name: str) -> Optional[str]:
        """Chuyển đổi tên bệnh DermNet gốc sang tên chuẩn trong config.
        
        Ví dụ: 'Acne and Rosacea Photos' → 'Acne and Rosacea'
        """
        # Nếu tên đã khớp config, trả về ngay
        if raw_name in self.class_to_idx:
            return raw_name
        # Nếu không, tra bảng ánh xạ DermNet
        mapped = DERMNET_NAME_MAPPING.get(raw_name)
        if mapped and mapped in self.class_to_idx:
            return mapped
        return None
        
    def _load_samples(self):
        """Quét thư mục để lấy danh sách toàn bộ ảnh và nhãn tương ứng."""
        # Thử tìm metadata.csv để đọc cho thư mục phẳng (flat directory)
        meta_path = self.img_dir / "metadata.csv"
        if not meta_path.exists():
            meta_path = self.img_dir.parent / "metadata.csv"
            
        if meta_path.exists():
            df = pd.read_csv(meta_path)
            skipped = 0
            for _, row in df.iterrows():
                img_id = row['image_id']
                raw_class_name = row['diagnosis']
                
                # Ánh xạ tên bệnh DermNet → tên config
                class_name = self._normalize_class_name(raw_class_name)
                if class_name is None:
                    skipped += 1
                    continue
                
                img_path = self.img_dir / f"{img_id}.jpg"
                if not img_path.exists():
                    img_path = self.img_dir / f"{img_id}.png"
                if not img_path.exists():
                    continue
                    
                if self.is_binary:
                    label = 1 if class_name in self.high_risk else 0
                else:
                    label = self.class_to_idx[class_name]
                self.samples.append((str(img_path), label))
            
            if skipped > 0:
                print(f"  [Dataset] Bo qua {skipped} anh do ten benh khong khop config.")
            print(f"  [Dataset] Da nap thanh cong {len(self.samples)} anh tu metadata.csv")
            return
            
        # Cách cũ: Đọc theo thư mục con
        for class_name in self.classes:
            class_dir = self.img_dir / class_name
            if not class_dir.exists():
                continue
                
            for img_path in class_dir.glob("*.jpg"):
                if self.is_binary:
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
            raise ValueError(f"Khong the doc anh: {img_path_str}")
            
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
                
                # Encode 7 features từ text/số thô → số chuẩn hóa [0, 1]
                age = float(row.get('age', 50)) / 100.0
                sex = SEX_MAP.get(str(row.get('sex', 'male')), 0.5)
                anatom_site = SITE_MAP.get(str(row.get('anatom_site', '')), 0.0)
                duration = min(float(row.get('duration', 30)) / 365.0, 1.0)
                symptom = SYMPTOM_MAP.get(str(row.get('symptoms', 'none')), 0.0)
                skin_type = SKIN_TYPE_MAP.get(str(row.get('skin_type', 'III')), 0.4)
                family_history = FAMILY_HISTORY_MAP.get(str(row.get('family_history', 'no')), 0.0)
                
                clinical_features = torch.tensor([age, sex, anatom_site, duration, symptom, skin_type, family_history], dtype=torch.float32)
            else:
                # Nếu thiếu metadata, trả về vector 0
                clinical_features = torch.zeros(7, dtype=torch.float32)
                
            return image, clinical_features, label_tensor

        # Trả về FloatTensor cho features, tensor int64 cho label (Vision-only)
        return image, label_tensor
