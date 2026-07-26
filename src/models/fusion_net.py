import torch
import torch.nn as nn
from typing import Tuple

from .vision_branch import VisionBranch
from .clinical_branch import ClinicalBranch

class MultimodalFusionNet(nn.Module):
    """Mạng Đa phương thức (Multimodal Fusion).
    
    Kết hợp đặc trưng hình ảnh (Vision) và đặc trưng lâm sàng (Clinical) 
    để đưa ra dự đoán phân loại đa lớp.
    """
    
    def __init__(
        self, 
        vision_model_name: str = "tf_efficientnetv2_m.in21k_ft_in1k",
        num_classes: int = 23,
        clinical_in_features: int = 7,
        clinical_out_features: int = 64,
        pretrained: bool = True
    ):
        super().__init__()
        
        # 1. Vision Branch (đã được cấu hình để xuất feature thay vì logits)
        self.vision_branch = VisionBranch(
            model_name=vision_model_name,
            num_classes=num_classes, # không quan trọng vì ta sẽ extract feature
            pretrained=pretrained
        )
        # Lấy kích thước vector đặc trưng TRƯỚC KHI chuyển classifier thành Identity
        vision_out_features = self.vision_branch.backbone.get_classifier().in_features
        
        self.vision_branch.set_extract_features(True)
        
        # 2. Clinical Branch
        self.clinical_branch = ClinicalBranch(
            in_features=clinical_in_features,
            out_features=clinical_out_features
        )
        
        # 3. Fusion Head (Gộp 2 đặc trưng lại)
        fusion_dim = vision_out_features + clinical_out_features
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        
    def freeze_vision_branch(self):
        """Đóng băng Vision Branch (dùng cho Step 1 của 2-Step Training)."""
        for param in self.vision_branch.parameters():
            param.requires_grad = False
            
    def unfreeze_vision_branch(self):
        """Mở khóa Vision Branch (dùng cho Step 2 của 2-Step Training)."""
        for param in self.vision_branch.parameters():
            param.requires_grad = True

    def forward(self, img: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor hình ảnh, shape (B, 3, H, W)
            clinical: Tensor đặc trưng lâm sàng, shape (B, 5)
        Returns:
            Logits phân loại, shape (B, num_classes)
        """
        # Trích xuất đặc trưng hình ảnh (B, 1280)
        v_features = self.vision_branch(img)
        
        # Trích xuất đặc trưng lâm sàng (B, 64)
        c_features = self.clinical_branch(clinical)
        
        # Gộp đặc trưng (B, 1344)
        fused = torch.cat((v_features, c_features), dim=1)
        
        # Đưa qua Fusion Head để ra dự đoán (B, 7)
        logits = self.fusion_head(fused)
        
        return logits
