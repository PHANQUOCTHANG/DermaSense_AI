import pytest
import torch
import pandas as pd
from pathlib import Path
import tempfile
import cv2
import os

from src.models.fusion_net import MultimodalFusionNet
from src.data.dataset import DermDataset
from src.models.clinical_branch import ClinicalBranch

def test_clinical_branch_output_shape():
    batch_size = 4
    in_features = 5
    out_features = 64
    
    model = ClinicalBranch(in_features=in_features, out_features=out_features)
    x = torch.randn(batch_size, in_features)
    
    output = model(x)
    assert output.shape == (batch_size, out_features)

def test_fusion_net_forward_pass():
    batch_size = 2
    num_classes = 7
    img_size = 384
    
    # Sử dụng version cực nhỏ của EfficientNet (B0) để test nhanh, tránh tải nặng
    model = MultimodalFusionNet(
        vision_model_name="tf_efficientnet_b0", 
        num_classes=num_classes,
        clinical_in_features=5,
        clinical_out_features=64,
        pretrained=False
    )
    
    images = torch.randn(batch_size, 3, img_size, img_size)
    clinical = torch.randn(batch_size, 5)
    
    # Step 1: test freeze
    model.freeze_vision_branch()
    # Kiểm tra xem vision branch có bị đóng băng không
    vision_requires_grad = any(p.requires_grad for p in model.vision_branch.parameters())
    assert not vision_requires_grad, "Vision branch chưa được đóng băng"
    
    logits = model(images, clinical)
    assert logits.shape == (batch_size, num_classes)
    
    # Step 2: test unfreeze
    model.unfreeze_vision_branch()
    vision_requires_grad = any(p.requires_grad for p in model.vision_branch.parameters())
    assert vision_requires_grad, "Vision branch chưa được mở khóa"

def test_dermdataset_multimodal():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Tạo thư mục chứa class
        class_dir = tmp_path / "MEL"
        class_dir.mkdir(parents=True)
        
        # Tạo ảnh giả
        img_path = class_dir / "ISIC_12345.jpg"
        dummy_img = torch.randint(0, 255, (384, 384, 3), dtype=torch.uint8).numpy()
        cv2.imwrite(str(img_path), dummy_img)
        
        # Tạo metadata giả
        csv_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "image_id": ["ISIC_12345"],
            "age": [0.5],
            "sex": [1.0],
            "anatom_site": [0.2],
            "duration": [0.0],
            "symptoms": [0.0]
        })
        df.to_csv(csv_path, index=False)
        
        dataset = DermDataset(
            img_dir=str(tmp_path),
            is_binary=False,
            use_clinical=True,
            clinical_csv=str(csv_path)
        )
        
        assert len(dataset) == 1
        
        # Lấy phần tử
        image, clinical, label = dataset[0]
        
        assert image.shape == (384, 384, 3) # OpenCV format chưa được transform
        assert clinical.shape == (5,)
        assert clinical[0].item() == 0.5
        assert clinical[1].item() == 1.0
        assert isinstance(label.item(), int)
