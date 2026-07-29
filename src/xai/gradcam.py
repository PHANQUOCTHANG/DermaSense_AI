"""
DermaSense AI — Grad-CAM (Gradient-weighted Class Activation Mapping)
Module tạo bản đồ nhiệt (heatmap) giải thích quyết định của AI.

Grad-CAM hoạt động bằng cách:
1. Hook vào lớp convolutional cuối cùng của backbone.
2. Forward pass để lấy logits.
3. Backward pass từ class target để lấy gradient.
4. Tính trung bình gradient theo spatial dimensions → trọng số.
5. Nhân trọng số với feature maps → heatmap.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class GradCAM:
    """Grad-CAM engine cho MultimodalFusionNet.
    
    Tự động hook vào lớp conv cuối cùng của EfficientNetV2 backbone
    bên trong MultimodalFusionNet.vision_branch.backbone.
    """
    
    def __init__(self, model, target_layer=None):
        """
        Args:
            model: MultimodalFusionNet đã load trọng số.
            target_layer: Lớp conv cần hook. Nếu None, tự động chọn lớp cuối.
        """
        self.model = model
        self.model.eval()
        
        # Tự động tìm lớp conv cuối cùng trong EfficientNetV2
        if target_layer is None:
            # EfficientNetV2 trong timm: backbone.conv_head hoặc lớp cuối trong blocks
            backbone = model.vision_branch.backbone
            if hasattr(backbone, 'conv_head'):
                self.target_layer = backbone.conv_head
            else:
                # Fallback: lấy block cuối cùng
                blocks = list(backbone.children())
                self.target_layer = blocks[-3]  # Thường là layer trước global pool
        else:
            self.target_layer = target_layer
            
        # Biến lưu trữ feature maps và gradients
        self.feature_maps = None
        self.gradients = None
        
        # Đăng ký hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Đăng ký forward hook và backward hook vào target layer."""
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
    def generate(
        self, 
        image_tensor: torch.Tensor, 
        clinical_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Tạo heatmap Grad-CAM.
        
        Args:
            image_tensor: Tensor ảnh đã qua transforms, shape (1, 3, H, W).
            clinical_tensor: Tensor đặc trưng lâm sàng, shape (1, 7).
            target_class: Class index cần giải thích. Nếu None, dùng predicted class.
            
        Returns:
            Heatmap numpy array, shape (H, W), giá trị [0, 1].
        """
        # Bật gradient tracking
        self.model.zero_grad()
        image_tensor.requires_grad_(True)
        
        # Forward pass
        logits = self.model(image_tensor, clinical_tensor)
        
        # Xác định target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
            
        # Backward pass từ target class score
        target_score = logits[0, target_class]
        target_score.backward(retain_graph=True)
        
        # Lấy gradient và feature maps
        gradients = self.gradients  # (1, C, h, w)
        feature_maps = self.feature_maps  # (1, C, h, w)
        
        # Tính trọng số: Global Average Pooling trên gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Nhân trọng số với feature maps rồi cộng lại
        cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)  # (1, 1, h, w)
        
        # ReLU: chỉ giữ ảnh hưởng tích cực
        cam = F.relu(cam)
        
        # Normalize về [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def generate_overlay(
        self,
        original_image: np.ndarray,
        image_tensor: torch.Tensor,
        clinical_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Tạo ảnh overlay heatmap lên ảnh gốc.
        
        Args:
            original_image: Ảnh gốc RGB, shape (H, W, 3), dtype uint8.
            image_tensor: Tensor ảnh đã transforms.
            clinical_tensor: Tensor đặc trưng lâm sàng.
            target_class: Class index cần giải thích.
            alpha: Độ trong suốt của heatmap (0=ảnh gốc, 1=chỉ heatmap).
            colormap: OpenCV colormap (mặc định JET).
            
        Returns:
            Tuple gồm:
            - overlay_image: Ảnh RGB với heatmap overlay, shape (H, W, 3).
            - heatmap_colored: Heatmap đã tô màu, shape (H, W, 3).
            - intensity_score: Điểm cường độ tập trung (0-1), càng cao = AI càng tập trung vào 1 vùng.
        """
        # Tạo heatmap thô
        cam = self.generate(image_tensor, clinical_tensor, target_class)
        
        # Resize heatmap về kích thước ảnh gốc
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Áp dụng colormap
        heatmap_uint8 = np.uint8(255 * cam_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap lên ảnh gốc
        overlay = np.float32(original_image) * (1 - alpha) + np.float32(heatmap_colored) * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # Tính điểm cường độ tập trung (Focality Score)
        # Dùng phần trăm diện tích có activation > 50% max
        high_activation_ratio = np.mean(cam_resized > 0.5)
        # Focality cao = vùng kích hoạt nhỏ gọn = AI tập trung vào 1 điểm rõ ràng
        intensity_score = 1.0 - min(high_activation_ratio * 3, 1.0)
        
        return overlay, heatmap_colored, intensity_score


def get_top_activated_regions(
    cam: np.ndarray, 
    original_image: np.ndarray,
    threshold: float = 0.6
) -> list:
    """Tìm các vùng kích hoạt mạnh nhất trên heatmap.
    
    Args:
        cam: Heatmap thô từ generate(), shape (h, w).
        original_image: Ảnh gốc để lấy kích thước.
        threshold: Ngưỡng kích hoạt (0-1).
        
    Returns:
        Danh sách dict chứa thông tin từng vùng: {x, y, w, h, intensity}.
    """
    h, w = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Tạo binary mask
    mask = (cam_resized > threshold).astype(np.uint8) * 255
    
    # Tìm contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        area = rw * rh
        if area > 100:  # Lọc bỏ vùng quá nhỏ
            intensity = float(np.mean(cam_resized[y:y+rh, x:x+rw]))
            regions.append({
                'x': x, 'y': y, 'w': rw, 'h': rh,
                'intensity': intensity,
                'area_percent': (area / (h * w)) * 100
            })
    
    # Sắp xếp theo cường độ giảm dần
    regions.sort(key=lambda r: r['intensity'], reverse=True)
    return regions
