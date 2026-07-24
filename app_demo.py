import streamlit as st
import cv2
import numpy as np
import yaml
from PIL import Image
from src.data.preprocessing import preprocess_single_image
import tempfile
import os

st.set_page_config(page_title="DermaSense AI - Demo Stage 1", layout="wide")

st.title("🔬 DermaSense AI - Demo Tiền Xử Lý (Stage 1)")
st.markdown("""
Tải lên một bức ảnh da liễu để kiểm tra hệ thống màng lọc của chúng tôi:
1. **Skin Detection**: Loại bỏ ảnh không phải da.
2. **Blur Detection**: Loại bỏ ảnh quá mờ.
3. **Hair Removal (DullRazor)**: Xóa lông che khuất tổn thương.
4. **Color Balance (Gray World)**: Cân bằng trắng.
""")

# Load config
def load_config():
    with open("configs/stage1_preprocessing.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

uploaded_file = st.file_uploader("Chọn một ảnh da liễu...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh Gốc")
        st.image(img_rgb, use_container_width=True)
        
        if st.button("Chạy Tiền Xử Lý 🚀", type="primary", use_container_width=True):
            with st.spinner("Đang xử lý ảnh..."):
                # Cần lưu ảnh ra file tạm vì preprocess_single_image nhận đường dẫn file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    cv2.imwrite(tmp.name, img_bgr)
                    tmp_path = tmp.name
                
                try:
                    processed_img, is_blurry, is_skin = preprocess_single_image(tmp_path, config["preprocessing"])
                    
                    with col2:
                        st.subheader("Kết quả")
                        
                        if not is_skin:
                            st.error("🚨 **BỊ LOẠI:** Ảnh không phải ảnh da (hoặc diện tích da quá nhỏ).")
                            st.info("💡 Hệ thống nhận diện màu da (HSV/YCrCb) kết luận bức ảnh này không phù hợp để chẩn đoán.")
                        elif is_blurry:
                            st.warning("🚨 **BỊ LOẠI:** Ảnh quá mờ.")
                            st.info("💡 Phương sai Laplacian thấp hơn ngưỡng cho phép. Cần ảnh sắc nét hơn để bác sĩ/AI nhìn rõ tổn thương.")
                        else:
                            st.success("✅ **ĐƯỢC CHẤP NHẬN!**")
                            if processed_img is not None:
                                processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                                st.image(processed_rgb, caption="Ảnh sau khi làm sạch (Đã xóa lông, cân bằng màu & resize)", use_container_width=True)
                finally:
                    os.remove(tmp_path)
