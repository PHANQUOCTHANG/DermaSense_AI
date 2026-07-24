"""
DermaSense AI — Vision Branch
Module chứa mô hình thị giác máy tính (EfficientNetV2).
"""
import timm
import torch
import torch.nn as nn

class VisionBranch(nn.Module):
    def __init__(self, model_name: str = "tf_efficientnetv2_s.in21k_ft_in1k", num_classes: int = 2, pretrained: bool = True, drop_rate: float = 0.2):
        """Khởi tạo Vision Branch.
        
        Args:
            model_name: Tên mô hình trong thư viện timm (mặc định là EfficientNetV2-S).
            num_classes: Số lượng lớp cần phân loại (2 cho Stage A, 7 cho Stage B nếu dùng vision-only).
            pretrained: Sử dụng trọng số pre-trained trên ImageNet.
            drop_rate: Tỷ lệ Dropout cho classifier head.
        """
        super().__init__()
        
        # Load mô hình từ timm
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            drop_rate=drop_rate
        )
        
        # Lấy kích thước vector đặc trưng ở lớp cuối cùng
        in_features = self.backbone.get_classifier().in_features
        
        # Thay thế Classifier Head
        # nn.Identity() nếu muốn lấy features (cho Stage B)
        # nn.Linear() nếu muốn output trực tiếp (cho Stage A)
        self.backbone.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Thực thi qua mô hình.
        
        Args:
            x: Tensor ảnh (B, C, H, W)
            
        Returns:
            Logits Tensor (B, num_classes)
        """
        return self.backbone(x)
        
    def freeze_backbone(self):
        """Đóng băng trọng số của backbone (chỉ train classifier)."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
                
    def unfreeze_backbone(self):
        """Mở khóa trọng số toàn bộ mô hình để fine-tune."""
        for param in self.parameters():
            param.requires_grad = True
