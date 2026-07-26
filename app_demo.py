import streamlit as st
import cv2
import numpy as np
import yaml
import torch
from pathlib import Path
import tempfile
import os

from src.data.preprocessing import preprocess_single_image
from src.data.augmentations import get_val_transforms
from src.models.vision_branch import VisionBranch
from src.models.fusion_net import MultimodalFusionNet


st.set_page_config(page_title="DermaSense AI - Demo", layout="wide")

st.title("🔬 DermaSense AI - Hệ Thống Chẩn Đoán Ung Thư Da")
st.markdown("""
Hệ thống AI Đa phương thức kết hợp Hình ảnh da liễu và Bệnh án lâm sàng.
""")

# Load configs
@st.cache_resource
def load_configs():
    with open("configs/stage1_preprocessing.yaml", "r", encoding="utf-8") as f:
        stage1_cfg = yaml.safe_load(f)
    with open("configs/stage2_train_screener.yaml", "r", encoding="utf-8") as f:
        stage2_cfg = yaml.safe_load(f)
    with open("configs/stage3_train_multimodal.yaml", "r", encoding="utf-8") as f:
        stage3_cfg = yaml.safe_load(f)
    return stage1_cfg, stage2_cfg, stage3_cfg

stage1_config, stage2_config, stage3_config = load_configs()

# Load models
@st.cache_resource
def load_models(cfg2, cfg3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Stage 2 Model (Screener)
    model2 = VisionBranch(
        model_name=cfg2["model"]["name"],
        num_classes=2,
        pretrained=False
    ).to(device)
    ckpt2_path = Path("models/checkpoints/stage_a/best_model.pt")
    if ckpt2_path.exists():
        model2.load_state_dict(torch.load(ckpt2_path, map_location=device, weights_only=False)['model_state_dict'])
        model2.eval()
    else:
        model2 = None

    # Stage 3 Model (Multimodal)
    model3 = MultimodalFusionNet(
        vision_model_name=cfg3["model"]["vision_name"],
        num_classes=cfg3["model"]["num_classes"],
        clinical_in_features=cfg3["model"]["clinical_in_features"],
        pretrained=False
    ).to(device)
    ckpt3_path = Path("models/checkpoints/stage_b/best_model.pt")
    if ckpt3_path.exists():
        model3.load_state_dict(torch.load(ckpt3_path, map_location=device, weights_only=False)['model_state_dict'])
        model3.eval()
    else:
        model3 = None
        
    return model2, model3, device

model2, model3, device = load_models(stage2_config, stage3_config)

# --- SIDEBAR: Thông tin Bệnh án ---
st.sidebar.header("📋 Thông Tin Lâm Sàng")
st.sidebar.markdown("*(Dùng cho chẩn đoán chuyên sâu - Stage 3)*")
age = st.sidebar.slider("Tuổi", 0, 100, 45)
sex = st.sidebar.selectbox("Giới tính", ["male", "female"])
site = st.sidebar.selectbox("Vị trí tổn thương", ["head/neck", "anterior torso", "posterior torso", "upper extremity", "lower extremity", "palms/soles"])
duration = st.sidebar.number_input("Thời gian xuất hiện (ngày)", 0, 3650, 30)
symptom = st.sidebar.selectbox("Triệu chứng", ["none", "itch", "pain", "bleeding"])
skin_type = st.sidebar.selectbox("Loại da (Fitzpatrick)", ["I", "II", "III", "IV", "V", "VI"])
family_history = st.sidebar.selectbox("Tiền sử gia đình mắc bệnh da liễu", ["no", "yes"])

# Encode metadata
age_encoded = age / 100.0
sex_encoded = 1.0 if sex == "male" else 0.0
site_dict = {"head/neck": 0.0, "anterior torso": 0.2, "posterior torso": 0.4, "upper extremity": 0.6, "lower extremity": 0.8, "palms/soles": 1.0}
symp_dict = {"none": 0.0, "itch": 0.3, "pain": 0.6, "bleeding": 1.0}
skin_type_dict = {"I": 0.0, "II": 0.2, "III": 0.4, "IV": 0.6, "V": 0.8, "VI": 1.0}
fh_encoded = 1.0 if family_history == "yes" else 0.0

clinical_features = torch.tensor([[age_encoded, sex_encoded, site_dict[site], duration/365.0, symp_dict[symptom], skin_type_dict[skin_type], fh_encoded]], dtype=torch.float32)

# --- MAIN AREA ---
uploaded_file = st.file_uploader("Tải lên ảnh da liễu...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh Gốc")
        st.image(img_rgb, width='stretch')
        
        if st.button("Chạy Phân Tích AI 🚀", type="primary", width='stretch'):
            with st.spinner("Hệ thống đang xử lý toàn bộ luồng..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    cv2.imwrite(tmp.name, img_bgr)
                    tmp_path = tmp.name
                
                try:
                    # ==========================================
                    # BƯỚC 1: TIỀN XỬ LÝ (STAGE 1)
                    # ==========================================
                    processed_img, is_blurry, is_skin = preprocess_single_image(tmp_path, stage1_config["preprocessing"])
                    
                    with col2:
                        st.subheader("Tiến trình Phân Tích")
                        
                        if not is_skin:
                            st.error("🚨 **LOẠI BỎ:** Hình ảnh không chứa vùng da người.")
                        elif is_blurry and not stage1_config["preprocessing"]["blur_detection"]["flag_only"]:
                            st.warning("🚨 **LOẠI BỎ:** Ảnh quá mờ, không đạt tiêu chuẩn y khoa.")
                        else:
                            st.success("✅ **BƯỚC 1 - TIỀN XỬ LÝ:** Đạt chuẩn. Đã lọc nhiễu và cân bằng màu.")
                            if processed_img is not None:
                                processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                                # st.image(processed_rgb, caption="Ảnh sau tiền xử lý", width='stretch')
                                
                                val_transforms = get_val_transforms()
                                tensor_img = val_transforms(image=processed_rgb)["image"].unsqueeze(0).to(device)
                                
                                # ==========================================
                                # BƯỚC 2: SÀNG LỌC NHANH (STAGE 2)
                                # ==========================================
                                st.markdown("---")
                                st.markdown("### 🛡️ BƯỚC 2: SÀNG LỌC NHANH (Screener)")
                                if model2 is not None:
                                    with torch.no_grad():
                                        probs2 = torch.softmax(model2(tensor_img), dim=1)[0]
                                        prob_high = probs2[1].item()
                                    
                                    st.write(f"**Nguy cơ ác tính:** {prob_high:.1%}")
                                    st.progress(prob_high)
                                    
                                    if prob_high > 0.5:
                                        st.error("⚠️ **Phát hiện dấu hiệu rủi ro cao! Cần chẩn đoán chuyên sâu.**")
                                    else:
                                        st.success("✅ **Đánh giá sơ bộ: Lành tính.**")
                                else:
                                    st.warning("⚠️ Chưa có trọng số cho Stage 2.")
                                    
                                # ==========================================
                                # BƯỚC 3: CHẨN ĐOÁN CHUYÊN SÂU (STAGE 3)
                                # ==========================================
                                st.markdown("---")
                                st.markdown("### 🧠 BƯỚC 3: CHẨN ĐOÁN ĐA PHƯƠNG THỨC")
                                st.caption("Kết hợp Hình ảnh + Bệnh án")
                                
                                if model3 is not None:
                                    with torch.no_grad():
                                        logits3 = model3(tensor_img, clinical_features.to(device))
                                        probs3 = torch.softmax(logits3, dim=1)[0]
                                        
                                    classes = [
                                        "Mụn trứng cá & Mụn đỏ (Acne/Rosacea)", "Ung thư da / Dày sừng (AK/Cancer)", "Viêm da cơ địa (Atopic Dermatitis)", 
                                        "Bệnh bọng nước (Bullous Disease)", "Viêm mô tế bào / Chốc lở (Bacterial)", "Chàm (Eczema)", 
                                        "Phát ban dị ứng thuốc (Exanthems)", "Rụng tóc (Hair Loss)", "Herpes / HPV", 
                                        "Bệnh sắc tố (Nám/Bạch biến)", "Lupus / Bệnh mô liên kết", 
                                        "Ung thư hắc tố / Nốt ruồi (Melanoma/Nevi)", "Nấm móng (Nail Fungus)", "Viêm da tiếp xúc (Contact Dermatitis)", 
                                        "Vảy nến / Lichen phẳng (Psoriasis)", "Ghẻ / Ve cắn (Scabies/Lyme)", 
                                        "U lành tính (Benign Tumors)", "Bệnh hệ thống (Systemic)", 
                                        "Hắc lào / Nấm da (Tinea)", "Mề đay (Urticaria Hives)", "U mạch máu (Vascular Tumors)", 
                                        "Viêm mạch máu (Vasculitis)", "Mụn cóc / U mềm (Warts/Molluscum)"
                                    ]
                                    
                                    # Hiển thị Top 3
                                    top3_prob, top3_idx = torch.topk(probs3, 3)
                                    for i in range(3):
                                        cls_name = classes[top3_idx[i]]
                                        p = top3_prob[i].item()
                                        
                                        # Gán màu tùy theo bệnh để giao diện sinh động
                                        if "Ung thư" in cls_name or "Lupus" in cls_name or "Mạch máu" in cls_name or "Bọng nước" in cls_name or "Hệ thống" in cls_name:
                                            icon = "🔴"
                                        elif "Chàm" in cls_name or "Viêm" in cls_name or "Vảy nến" in cls_name or "Herpes" in cls_name or "Ghẻ" in cls_name:
                                            icon = "🟡"
                                        else:
                                            icon = "🟢"
                                            
                                        st.write(f"**{icon} {cls_name}:** {p:.1%}")
                                        st.progress(p)
                                        
                                    # Phần khuyến nghị
                                    st.markdown("---")
                                    st.markdown("### 💊 Khuyến Nghị Sơ Bộ")
                                    st.info("""
                                    *Lưu ý: Đây chỉ là tư vấn tham khảo từ AI, không thay thế chẩn đoán y khoa chính thức.*
                                    
                                    **Dựa trên dự đoán cao nhất:**
                                    - **Nếu là bệnh nhẹ (🟢):** Có thể tự theo dõi, giữ vệ sinh sạch sẽ, dùng thuốc bôi ngoài da không kê đơn (ví dụ: trị mụn, trị nấm).
                                    - **Nếu là bệnh trung bình (🟡):** Nên dưỡng ẩm da, tránh gãi hoặc tiếp xúc dị ứng nguyên. Nếu kéo dài quá 2 tuần, hãy thăm khám bác sĩ.
                                    - **Nếu là bệnh nặng (🔴):** Tuyệt đối KHÔNG tự ý bôi thuốc. Vui lòng đặt lịch khám ngay với Bác sĩ chuyên khoa Da Liễu để sinh thiết hoặc có phác đồ điều trị.
                                    """)
                                else:
                                    st.warning("⚠️ Chưa có trọng số cho Stage 3.")
                                    
                finally:
                    os.remove(tmp_path)
