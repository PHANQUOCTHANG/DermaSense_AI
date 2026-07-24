import streamlit as st
import cv2
import numpy as np
import yaml
import torch
from pathlib import Path
from src.data.preprocessing import preprocess_single_image
from src.data.augmentations import get_val_transforms
from src.models.vision_branch import VisionBranch
import tempfile
import os

st.set_page_config(page_title="DermaSense AI - Demo", layout="wide")

st.title("🔬 DermaSense AI - Demo Sàng Lọc Ung Thư Da")
st.markdown("""
Tải lên một bức ảnh da liễu để trải nghiệm quy trình chuẩn y khoa:
1. **Stage 1 (Tiền xử lý)**: Lọc ảnh rác, xóa lông, cân bằng trắng.
2. **Stage 2 (Phân loại AI)**: Phát hiện nguy cơ ung thư (Melanoma, BCC, AK) vs Lành tính (Nevus, VASC...).
""")

# Load configs
@st.cache_resource
def load_configs():
    with open("configs/stage1_preprocessing.yaml", "r", encoding="utf-8") as f:
        stage1_cfg = yaml.safe_load(f)
    with open("configs/stage2_train_screener.yaml", "r", encoding="utf-8") as f:
        stage2_cfg = yaml.safe_load(f)
    return stage1_cfg, stage2_cfg

stage1_config, stage2_config = load_configs()

# Load model
@st.cache_resource
def load_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionBranch(
        model_name=cfg["model"]["name"],
        num_classes=2,
        pretrained=False
    ).to(device)
    
    ckpt_path = Path("models/checkpoints/stage_a/best_model.pt")
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, device
    return None, device

model, device = load_model(stage2_config)

uploaded_file = st.file_uploader("Chọn một ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh Gốc")
        st.image(img_rgb, width='stretch')
        
        if st.button("Chạy Phân Tích AI 🚀", type="primary", width='stretch'):
            with st.spinner("Đang xử lý ảnh và chạy AI..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    cv2.imwrite(tmp.name, img_bgr)
                    tmp_path = tmp.name
                
                try:
                    # Stage 1
                    processed_img, is_blurry, is_skin = preprocess_single_image(tmp_path, stage1_config["preprocessing"])
                    
                    with col2:
                        st.subheader("Kết Quả Phân Tích")
                        
                        if not is_skin:
                            st.error("🚨 **BỊ LOẠI:** Hệ thống phát hiện đây không phải là ảnh chụp da.")
                        elif is_blurry and not stage1_config["preprocessing"]["blur_detection"]["flag_only"]:
                            st.warning("🚨 **BỊ LOẠI:** Ảnh quá mờ, bác sĩ và AI đều không thể chẩn đoán chính xác.")
                        else:
                            st.success("✅ **BƯỚC 1:** Tiền xử lý hoàn tất (Đã xóa lông, chuẩn màu).")
                            if processed_img is not None:
                                processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                                st.image(processed_rgb, caption="Ảnh sau tiền xử lý", width='stretch')
                                
                                # Stage 2
                                if model is not None:
                                    st.info("🧠 **BƯỚC 2:** AI đang đánh giá nguy cơ...")
                                    
                                    val_transforms = get_val_transforms()
                                    tensor_img = val_transforms(image=processed_rgb)["image"].unsqueeze(0).to(device)
                                    
                                    with torch.no_grad():
                                        logits = model(tensor_img)
                                        probs = torch.softmax(logits, dim=1)[0]
                                        
                                    prob_low = probs[0].item()
                                    prob_high = probs[1].item()
                                    
                                    st.write(f"**Xác suất Lành tính:** {prob_low:.1%}")
                                    st.progress(prob_low)
                                    
                                    st.write(f"**Xác suất Rủi ro cao:** {prob_high:.1%}")
                                    st.progress(prob_high)
                                    
                                    if prob_high > 0.5:
                                        st.error("⚠️ **CẢNH BÁO:** Phát hiện dấu hiệu rủi ro cao. Khuyến nghị đi khám bác sĩ ngay!")
                                    else:
                                        st.success("✅ **KẾT LUẬN:** Tổn thương có vẻ lành tính. Hãy tiếp tục theo dõi.")
                                else:
                                    st.warning("⚠️ Chưa tìm thấy file trọng số AI (.pt). Hãy huấn luyện mô hình trước!")
                finally:
                    os.remove(tmp_path)
