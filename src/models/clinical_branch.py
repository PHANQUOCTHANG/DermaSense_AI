import torch
import torch.nn as nn

class ClinicalBranch(nn.Module):
    """Mạng MLP xử lý dữ liệu lâm sàng (metadata).
    
    Chuyển đổi vector đặc trưng rời rạc (tuổi, giới tính, vị trí...) 
    thành một vector embedding dày đặc (dense representation).
    """
    
    def __init__(self, in_features: int = 7, out_features: int = 64):
        """
        Args:
            in_features: Số lượng đặc trưng đầu vào (mặc định 7: age, sex, site, duration, symptoms, skin_type, family_history).
            out_features: Kích thước của vector đầu ra.
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            nn.Linear(128, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor đặc trưng lâm sàng, shape (Batch, in_features)
        Returns:
            Tensor đã xử lý, shape (Batch, out_features)
        """
        return self.mlp(x)
