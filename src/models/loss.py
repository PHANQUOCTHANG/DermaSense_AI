"""
DermaSense AI — Loss Functions
Chứa các hàm mất mát tùy chỉnh, bao gồm Focal Loss để giải quyết vấn đề mất cân bằng dữ liệu.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """Khởi tạo Binary Focal Loss.
        
        Phương trình: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        Giúp mô hình tập trung vào các mẫu khó đoán (hard examples) 
        và giảm bớt trọng số của các mẫu dễ đoán (thuộc lớp chiếm đa số).
        
        Args:
            alpha: Trọng số cân bằng lớp (thường gán cho lớp thiểu số / Positive class).
                   Nếu dùng WeightedRandomSampler, có thể để mặc định hoặc None.
            gamma: Tham số focus (thường = 2.0). Gamma càng lớn thì càng tập trung vào mẫu khó.
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits từ mô hình (shape: [batch_size, 2])
            targets: Nhãn thực tế (shape: [batch_size]) (chứa 0 hoặc 1)
        """
        # Chuyển logits thành xác suất bằng softmax
        probs = F.softmax(inputs, dim=1)
        
        # Lấy xác suất của lớp đúng (p_t)
        # Gather theo dim=1, index là targets (shape: [batch_size, 1])
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Tính log(p_t) - dùng log_softmax để ổn định số học
        log_probs = F.log_softmax(inputs, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Tính Focal Loss
        loss = - (1 - pt) ** self.gamma * log_pt
        
        # Áp dụng alpha (nếu có)
        if self.alpha is not None:
            # Nếu target = 1 thì nhân alpha, nếu target = 0 thì nhân (1-alpha)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
