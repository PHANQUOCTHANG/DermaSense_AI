"""
DermaSense AI — Augmentations Utility
Định nghĩa các phép biến đổi ảnh để làm giàu dữ liệu huấn luyện.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(config_aug: dict, img_size: int = 384) -> A.Compose:
    """Tạo pipeline augmentation cho tập Train.
    
    Args:
        config_aug: Cấu hình augmentation từ base_config hoặc stage2_balancing.
        img_size: Kích thước ảnh sau khi resize
        
    Returns:
        Albumentations Compose object.
    """
    basic_cfg = config_aug.get("basic", {})
    advanced_cfg = config_aug.get("advanced", {})
    
    transforms = [
        A.Resize(img_size, img_size)
    ]
    
    # 1. Augmentation cơ bản (áp dụng cho mọi ảnh)
    if basic_cfg.get("horizontal_flip", True):
        transforms.append(A.HorizontalFlip(p=0.5))
    if basic_cfg.get("vertical_flip", True):
        transforms.append(A.VerticalFlip(p=0.5))
    if basic_cfg.get("random_rotate_90", True):
        transforms.append(A.RandomRotate90(p=0.5))
        
    bc_cfg = basic_cfg.get("brightness_contrast", {})
    if bc_cfg:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=bc_cfg.get("brightness_limit", 0.2),
                contrast_limit=bc_cfg.get("contrast_limit", 0.2),
                p=0.5
            )
        )
        
    # 2. Augmentation nâng cao (Áp dụng chung, nhưng sẽ cấu hình riêng ở sampler nếu cần)
    if advanced_cfg.get("enabled", False):
        if advanced_cfg.get("color_jitter", {}):
            cj = advanced_cfg["color_jitter"]
            transforms.append(
                A.HueSaturationValue(
                    hue_shift_limit=cj.get("hue_shift", 10),
                    sat_shift_limit=cj.get("sat_shift", 20),
                    val_shift_limit=cj.get("val_shift", 20),
                    p=0.3
                )
            )
        
        if advanced_cfg.get("grid_distortion", {}).get("enabled", False):
            gd = advanced_cfg["grid_distortion"]
            transforms.append(
                A.GridDistortion(
                    num_steps=gd.get("num_steps", 5),
                    distort_limit=gd.get("distort_limit", 0.3),
                    p=0.2
                )
            )
            
        # Thêm CoarseDropout (Random Erasing) để mô hình học context tốt hơn
        if advanced_cfg.get("coarse_dropout", True):
            transforms.append(
                A.CoarseDropout(
                    max_holes=8, 
                    max_height=32, 
                    max_width=32, 
                    min_holes=1, 
                    min_height=8, 
                    min_width=8, 
                    fill_value=0, 
                    p=0.5
                )
            )
            
    # Normalize và chuyển thành Tensor (Luôn áp dụng cuối cùng)
    # ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)

def get_val_transforms(img_size: int = 384) -> A.Compose:
    """Tạo pipeline augmentation cho tập Validation/Test.
    (Bao gồm Resize, Normalize và chuyển thành Tensor).
    
    Returns:
        Albumentations Compose object.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
