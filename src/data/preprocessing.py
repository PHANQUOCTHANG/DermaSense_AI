"""
DermaSense AI — Preprocessing Module
Chứa các thuật toán xử lý ảnh: Laplacian blur detection, DullRazor hair removal, Gray World balance, Resize.
"""
import cv2
import numpy as np


def compute_laplacian_variance(image: np.ndarray) -> float:
    """Tính phương sai của bộ lọc Laplacian để phát hiện ảnh mờ.
    
    Args:
        image: Ảnh BGR.
    
    Returns:
        variance: Phương sai Laplacian (float). < ngưỡng (vd: 100) -> ảnh mờ.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return float(variance)


def is_skin_image(image: np.ndarray, min_skin_ratio: float = 0.05) -> tuple:
    """Kiểm tra xem ảnh đầu vào có phải là ảnh da hay không bằng phân tích không gian màu.
    Sử dụng kết hợp YCrCb và HSV để định vị pixel màu da người.
    
    Args:
        image: Ảnh BGR.
        min_skin_ratio: Tỷ lệ phần trăm pixel da tối thiểu để được coi là ảnh da (0.05 = 5%).
        
    Returns:
        (is_skin, skin_ratio): bool, float
    """
    # Chuyển sang không gian màu HSV và YCrCb
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Ngưỡng màu da điển hình trong HSV
    # H: 0-20, S: 48-255, V: 80-255
    lower_hsv = np.array([0, 48, 80], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Ngưỡng màu da điển hình trong YCrCb
    # Y: 80-255, Cr: 133-173, Cb: 77-127
    lower_ycrcb = np.array([80, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    # Kết hợp 2 mask (phải thoả mãn cả 2 để chắc chắn là da)
    skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    
    # Tính tỷ lệ pixel da trên tổng số pixel
    total_pixels = image.shape[0] * image.shape[1]
    skin_pixels = cv2.countNonZero(skin_mask)
    skin_ratio = skin_pixels / total_pixels
    
    return skin_ratio >= min_skin_ratio, skin_ratio



def dullrazor_hair_removal(image: np.ndarray, kernel_size: int = 17, threshold: int = 10, inpaint_radius: int = 6) -> np.ndarray:
    """Xóa lông trên ảnh da liễu bằng thuật toán DullRazor (đơn giản hóa).
    
    Args:
        image: Ảnh đầu vào BGR.
        kernel_size: Kích thước kernel cho morphological black-hat.
        threshold: Ngưỡng nhị phân hóa để phát hiện lông.
        inpaint_radius: Bán kính inpaint (vá ảnh).
        
    Returns:
        Ảnh sau khi đã xóa lông (BGR).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Morphological Black-Hat để phát hiện các cấu trúc tối nhỏ (lông) trên nền sáng
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Gaussian Blur để giảm nhiễu trước khi nhị phân
    blurred = cv2.GaussianBlur(blackhat, (3, 3), 0)
    
    # Binary Threshold để tạo mask chỉ chứa lông
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    
    # Inpaint ảnh gốc bằng mask lông
    inpainted_image = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
    
    return inpainted_image


def gray_world_balance(image: np.ndarray) -> np.ndarray:
    """Cân bằng trắng theo giả thuyết Gray World (Thế giới xám).
    
    Args:
        image: Ảnh đầu vào BGR.
        
    Returns:
        Ảnh đã cân bằng trắng (BGR).
    """
    b, g, r = cv2.split(image)
    
    # Tính mean từng kênh
    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)
    
    # Tính mean tổng
    mean_avg = (mean_b + mean_g + mean_r) / 3
    
    if mean_avg == 0:
        return image
        
    # Scale từng kênh
    b_balanced = np.clip(b * (mean_avg / mean_b), 0, 255).astype(np.uint8)
    g_balanced = np.clip(g * (mean_avg / mean_g), 0, 255).astype(np.uint8)
    r_balanced = np.clip(r * (mean_avg / mean_r), 0, 255).astype(np.uint8)
    
    balanced_image = cv2.merge([b_balanced, g_balanced, r_balanced])
    return balanced_image


def resize_image(image: np.ndarray, target_size: tuple = (384, 384)) -> np.ndarray:
    """Resize ảnh về kích thước chuẩn. (Normalization thực hiện lúc train).
    
    Args:
        image: Ảnh đầu vào.
        target_size: (width, height) muốn resize.
        
    Returns:
        Ảnh đã resize.
    """
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return resized


def preprocess_single_image(image_path: str, config: dict) -> tuple:
    """Thực thi pipeline tiền xử lý cho 1 ảnh.
    
    Args:
        image_path: Đường dẫn ảnh gốc.
        config: Dictionary chứa tham số từ stage1_preprocessing.yaml.
        
    Returns:
        (processed_image, is_blurry, is_skin).
        processed_image: np.ndarray ảnh đã xử lý (hoặc None nếu không phải da/lỗi).
        is_blurry: bool. True nếu variance < threshold.
        is_skin: bool. True nếu là ảnh da.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, False
        
    # 1. Skin Detection (Kiểm tra xem có phải ảnh da không)
    skin_cfg = config.get("skin_detection", {})
    if skin_cfg.get("enabled", True):
        is_skin, skin_ratio = is_skin_image(image, min_skin_ratio=skin_cfg.get("min_skin_ratio", 0.15))
        if not is_skin:
            return None, False, False # (processed_img, is_blurry, is_skin)
            
    # 2. Blur Detection
    blur_cfg = config.get("blur_detection", {})
    variance = compute_laplacian_variance(image)
    is_blurry = variance < blur_cfg.get("threshold", 100.0)
    
    # 2. Hair Removal
    hair_cfg = config.get("hair_removal", {})
    if hair_cfg.get("enabled", True):
        image = dullrazor_hair_removal(
            image, 
            kernel_size=hair_cfg.get("kernel_size", 17),
            threshold=hair_cfg.get("threshold", 10),
            inpaint_radius=hair_cfg.get("inpaint_radius", 6)
        )
        
    # 3. Color Balance
    color_cfg = config.get("color_balance", {})
    if color_cfg.get("method", "gray_world") == "gray_world":
        image = gray_world_balance(image)
        
    # 4. Resize
    resize_cfg = config.get("resize", {})
    target_size = tuple(resize_cfg.get("target_size", [384, 384]))
    image = resize_image(image, target_size=target_size)
    
    return image, is_blurry, True
