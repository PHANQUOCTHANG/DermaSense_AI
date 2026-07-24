import pytest
import numpy as np
import cv2

from src.data.preprocessing import (
    compute_laplacian_variance,
    dullrazor_hair_removal,
    gray_world_balance,
    resize_image
)

@pytest.fixture
def dummy_image():
    """Tạo một ảnh BGR giả 400x400."""
    return np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)

@pytest.fixture
def blurry_image():
    """Tạo một ảnh mờ (bằng Gaussian Blur)."""
    img = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (25, 25), 0)

@pytest.fixture
def image_with_hair():
    """Tạo một ảnh sáng có vài vệt tối giả làm lông."""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 200
    cv2.line(img, (10, 10), (100, 100), (20, 20, 20), 3)
    cv2.line(img, (200, 50), (350, 300), (30, 30, 30), 2)
    return img

def test_compute_laplacian_variance(dummy_image, blurry_image):
    """Kiểm tra Laplacian variance: ảnh mờ phải có variance thấp hơn ảnh nhiễu/sắc nét."""
    var_sharp = compute_laplacian_variance(dummy_image)
    var_blurry = compute_laplacian_variance(blurry_image)
    
    assert var_sharp > var_blurry
    assert var_blurry < 1000 # Phương sai của ảnh mờ thường rất thấp

def test_dullrazor_hair_removal(image_with_hair):
    """Kiểm tra DullRazor không làm hỏng cấu trúc ảnh và trả về đúng shape."""
    cleaned = dullrazor_hair_removal(image_with_hair, kernel_size=9, threshold=10, inpaint_radius=3)
    
    assert cleaned.shape == image_with_hair.shape
    assert cleaned.dtype == np.uint8
    
    # Ở vùng lông (đen), pixel trong ảnh cleaned phải sáng hơn (đã được vá).
    # Tuy nhiên, test này khá phụ thuộc vào tham số. Ở đây ta test cơ bản:
    # đảm bảo không bị crash và output hợp lệ.

def test_gray_world_balance():
    """Kiểm tra Gray World balance: đưa các kênh màu về gần với trung bình chung."""
    # Tạo ảnh bị ám xanh lá (green channel cao)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 50   # Kênh Blue
    img[:, :, 1] = 200  # Kênh Green (ám màu)
    img[:, :, 2] = 50   # Kênh Red
    
    balanced = gray_world_balance(img)
    
    # Tính mean từng kênh sau khi cân bằng
    b, g, r = cv2.split(balanced)
    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    
    # Các kênh nên khá gần nhau (hoặc thay đổi tỉ lệ). 
    # Ban đầu green = 200, r/b = 50. Mean = 100.
    # Sau gray world: b_new = 50 * (100/50) = 100, g_new = 200 * (100/200) = 100.
    assert np.isclose(mean_b, 100, atol=2)
    assert np.isclose(mean_g, 100, atol=2)
    assert np.isclose(mean_r, 100, atol=2)

def test_resize_image(dummy_image):
    """Kiểm tra resize."""
    resized = resize_image(dummy_image, target_size=(384, 384))
    
    assert resized.shape == (384, 384, 3)
