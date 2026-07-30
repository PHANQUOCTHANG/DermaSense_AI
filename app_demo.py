"""
DermaSense AI — Giao diện Chẩn đoán Da liễu Chuyên nghiệp
Phiên bản 2.0: Tích hợp XAI (Grad-CAM) + UI Premium Dark Theme + Real-time Camera
"""
import streamlit as st
import cv2
import numpy as np
import yaml
import torch
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import tempfile
import os
import time
import threading
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from src.data.preprocessing import preprocess_single_image
from src.data.augmentations import get_val_transforms
from src.models.vision_branch import VisionBranch
from src.models.fusion_net import MultimodalFusionNet
from src.xai.gradcam import GradCAM, get_top_activated_regions
from src.xai.calibration import (
    compute_confidence_metrics, 
    format_star_display, 
    get_disease_risk_info
)

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="DermaSense AI — Hệ Thống Chẩn Đoán Da Liễu Thông Minh",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — DARK MEDICAL THEME
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ===== GLOBAL ===== */
    .stApp {
        background: linear-gradient(135deg, #0a0f1a 0%, #111827 50%, #0d1321 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1a1f2e 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] label {
        color: #94a3b8 !important;
    }
    
    /* ===== HEADER GRADIENT ===== */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #06b6d4, #3b82f6, #8b5cf6, #06b6d4);
        background-size: 200% 100%;
        animation: shimmer 3s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%, 100% { background-position: 200% 0; }
        50% { background-position: -200% 0; }
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin: 0;
        font-weight: 400;
    }
    
    /* ===== GLASSMORPHISM CARDS ===== */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        box-shadow: 0 8px 32px rgba(56, 189, 248, 0.08);
    }
    .glass-card h3 {
        color: #e2e8f0;
        font-weight: 700;
        margin-bottom: 12px;
    }
    
    /* ===== RESULT CARDS ===== */
    .result-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(16px);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 10px;
        border-left: 4px solid;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    .result-card.danger { border-left-color: #ff1744; }
    .result-card.warning { border-left-color: #ff9800; }
    .result-card.safe { border-left-color: #00e676; }
    
    .result-name {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 4px;
    }
    .result-prob {
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0;
    }
    .result-prob.danger { color: #ff1744; }
    .result-prob.warning { color: #ff9800; }
    .result-prob.safe { color: #00e676; }
    
    /* ===== PROGRESS BAR CUSTOM ===== */
    .progress-container {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
        margin: 6px 0 2px 0;
    }
    .progress-bar {
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .progress-bar.danger { background: linear-gradient(90deg, #ff1744, #ff5252); }
    .progress-bar.warning { background: linear-gradient(90deg, #ff9800, #ffc107); }
    .progress-bar.safe { background: linear-gradient(90deg, #00e676, #69f0ae); }
    
    /* ===== CONFIDENCE METER ===== */
    .confidence-meter {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
    }
    .confidence-stars {
        font-size: 2.2rem;
        letter-spacing: 4px;
        margin: 8px 0;
        filter: drop-shadow(0 0 8px rgba(255, 193, 7, 0.4));
    }
    .confidence-label {
        font-size: 1.1rem;
        font-weight: 700;
        margin: 4px 0 0 0;
    }
    .confidence-score {
        font-size: 0.85rem;
        color: #64748b;
    }
    
    /* ===== URGENCY BADGE ===== */
    .urgency-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .urgency-badge.urgent {
        background: rgba(255, 23, 68, 0.2);
        color: #ff1744;
        border: 1px solid rgba(255, 23, 68, 0.4);
    }
    .urgency-badge.warning {
        background: rgba(255, 152, 0, 0.2);
        color: #ff9800;
        border: 1px solid rgba(255, 152, 0, 0.4);
    }
    .urgency-badge.safe {
        background: rgba(0, 230, 118, 0.2);
        color: #00e676;
        border: 1px solid rgba(0, 230, 118, 0.4);
    }
    
    /* ===== SECTION DIVIDER ===== */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.3), transparent);
        margin: 20px 0;
        border: none;
    }
    
    /* ===== STEP INDICATOR ===== */
    .step-indicator {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
    }
    .step-number {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
        flex-shrink: 0;
    }
    .step-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .step-desc {
        font-size: 0.85rem;
        color: #64748b;
    }
    
    /* ===== XAI SECTION ===== */
    .xai-metric {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.1);
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
    }
    .xai-metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .xai-metric-label {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* ===== FADE-IN ANIMATION ===== */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeInUp 0.6s ease-out forwards;
    }
    .fade-in-delay-1 { animation-delay: 0.1s; }
    .fade-in-delay-2 { animation-delay: 0.2s; }
    .fade-in-delay-3 { animation-delay: 0.3s; }
    
    /* ===== PULSE ANIMATION cho cảnh báo ===== */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .pulse-alert {
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* ===== TABS STYLING ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(139, 92, 246, 0.3));
        color: #e2e8f0 !important;
    }
    
    /* ===== HIDE DEFAULT ELEMENTS ===== */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    
    /* ===== FILE UPLOADER ===== */
    .stFileUploader > div {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 2px dashed rgba(56, 189, 248, 0.25) !important;
        border-radius: 12px !important;
    }
    .stFileUploader > div:hover {
        border-color: rgba(56, 189, 248, 0.5) !important;
    }
    
    /* ===== BUTTON ===== */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.5) !important;
    }
    
    /* ===== RECOMMENDATION BOX ===== */
    .recommendation-box {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 12px;
    }
    .recommendation-box.danger {
        border-color: rgba(255, 23, 68, 0.3);
        background: rgba(255, 23, 68, 0.05);
    }
    .recommendation-box.warning {
        border-color: rgba(255, 152, 0, 0.3);
        background: rgba(255, 152, 0, 0.05);
    }
    .recommendation-box.safe {
        border-color: rgba(0, 230, 118, 0.3);
        background: rgba(0, 230, 118, 0.05);
    }
    .recommendation-title {
        font-size: 1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 8px;
    }
    .recommendation-text {
        font-size: 0.9rem;
        color: #cbd5e1;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero-header">
    <p class="hero-title">🔬 DermaSense AI</p>
    <p class="hero-subtitle">Hệ Thống Chẩn Đoán Da Liễu Đa Phương Thức — Kết hợp Hình ảnh Y khoa + Bệnh án Lâm sàng + Giải thích AI (XAI)</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD CONFIGS & MODELS
# ============================================================
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
        ckpt = torch.load(ckpt3_path, map_location=device, weights_only=False)
        model3.load_state_dict(ckpt['model_state_dict'])
        model3.eval()
    else:
        model3 = None
        
    return model2, model3, device

model2, model3, device = load_models(stage2_config, stage3_config)

# Khởi tạo Grad-CAM engine
gradcam_engine = None
if model3 is not None:
    try:
        gradcam_engine = GradCAM(model3)
    except Exception:
        gradcam_engine = None

# ============================================================
# DANH SÁCH 23 BỆNH
# ============================================================
CLASSES = [
    "Mụn trứng cá & Mụn đỏ (Acne/Rosacea)", 
    "Ung thư da / Dày sừng (AK/Cancer)", 
    "Viêm da cơ địa (Atopic Dermatitis)", 
    "Bệnh bọng nước (Bullous Disease)", 
    "Viêm mô tế bào / Chốc lở (Bacterial)", 
    "Chàm (Eczema)", 
    "Phát ban dị ứng thuốc (Exanthems)", 
    "Rụng tóc (Hair Loss)", 
    "Herpes / HPV", 
    "Bệnh sắc tố (Nám/Bạch biến)", 
    "Lupus / Bệnh mô liên kết", 
    "Ung thư hắc tố / Nốt ruồi (Melanoma/Nevi)", 
    "Nấm móng (Nail Fungus)", 
    "Viêm da tiếp xúc (Contact Dermatitis)", 
    "Vảy nến / Lichen phẳng (Psoriasis)", 
    "Ghẻ / Ve cắn (Scabies/Lyme)", 
    "U lành tính (Benign Tumors)", 
    "Bệnh hệ thống (Systemic)", 
    "Hắc lào / Nấm da (Tinea)", 
    "Mề đay (Urticaria Hives)", 
    "U mạch máu (Vascular Tumors)", 
    "Viêm mạch máu (Vasculitis)", 
    "Mụn cóc / U mềm (Warts/Molluscum)"
]


# ============================================================
# SIDEBAR — THÔNG TIN BỆNH ÁN
# ============================================================
with st.sidebar:
    st.markdown("## 📋 Hồ Sơ Lâm Sàng")
    st.markdown('<p style="color:#64748b; font-size:0.85rem;">Nhập thông tin bệnh nhân để AI phân tích đa phương thức</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    age = st.number_input("🎂 Tuổi", min_value=0, max_value=120, value=None, placeholder="Nhập tuổi...")
    sex = st.selectbox("⚧ Giới tính", ["male", "female"], index=None, placeholder="Chọn giới tính...", format_func=lambda x: "Nam" if x == "male" else "Nữ")
    site = st.selectbox("📍 Vị trí tổn thương", 
                        ["head/neck", "anterior torso", "posterior torso", "upper extremity", "lower extremity", "palms/soles"],
                        index=None, placeholder="Chọn vị trí...",
                        format_func=lambda x: {
                            "head/neck": "Đầu / Cổ",
                            "anterior torso": "Ngực / Bụng",
                            "posterior torso": "Lưng",
                            "upper extremity": "Tay",
                            "lower extremity": "Chân",
                            "palms/soles": "Lòng bàn tay / chân"
                        }[x])
    duration = st.number_input("⏱️ Thời gian xuất hiện (ngày)", min_value=0, max_value=3650, value=None, placeholder="Nhập số ngày...")
    symptom = st.selectbox("🩹 Triệu chứng", ["none", "itch", "pain", "bleeding"],
                           index=None, placeholder="Chọn triệu chứng...",
                           format_func=lambda x: {"none": "Không có", "itch": "Ngứa", "pain": "Đau", "bleeding": "Chảy máu"}[x])
    skin_type = st.selectbox("🎨 Loại da (Fitzpatrick)", ["I", "II", "III", "IV", "V", "VI"], index=None, placeholder="Chọn loại da...")
    family_history = st.selectbox("👨‍👩‍👧 Tiền sử gia đình", ["no", "yes"],
                                  index=None, placeholder="Chọn...",
                                  format_func=lambda x: "Không" if x == "no" else "Có")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:8px;">
        <p style="color:#475569; font-size:0.75rem; margin:0;">DermaSense AI v2.0</p>
        <p style="color:#334155; font-size:0.7rem; margin:2px 0 0 0;">EfficientNetV2-M + Clinical MLP</p>
    </div>
    """, unsafe_allow_html=True)

# Encode metadata sẽ được thực hiện khi người dùng bấm nút phân tích


# ============================================================
# HÀM TẠO BIỂU ĐỒ
# ============================================================
def create_radar_chart(probs, classes):
    """Tạo biểu đồ radar cho 23 bệnh."""
    # Lấy top 8 để biểu đồ không quá rối
    top_k = 8
    top_probs, top_indices = torch.topk(probs, top_k)
    
    labels = [classes[i] for i in top_indices]
    values = [p.item() * 100 for p in top_probs]
    
    # Đóng vòng radar
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill='toself',
        fillcolor='rgba(56, 189, 248, 0.15)',
        line=dict(color='#38bdf8', width=2),
        marker=dict(size=6, color='#38bdf8'),
        name='Xác suất (%)'
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(15, 23, 42, 0.6)',
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2],
                gridcolor='rgba(100, 116, 139, 0.2)',
                tickfont=dict(color='#64748b', size=9),
                ticksuffix='%'
            ),
            angularaxis=dict(
                gridcolor='rgba(100, 116, 139, 0.15)',
                tickfont=dict(color='#94a3b8', size=10),
            )
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=30, b=30, l=80, r=80),
        height=450,
    )
    
    return fig


def create_bar_chart(probs, classes):
    """Tạo biểu đồ thanh ngang cho tất cả 23 bệnh."""
    values = [probs[i].item() * 100 for i in range(len(classes))]
    
    # Sắp xếp giảm dần
    sorted_pairs = sorted(zip(classes, values), key=lambda x: x[1], reverse=True)
    sorted_labels = [p[0] for p in sorted_pairs]
    sorted_values = [p[1] for p in sorted_pairs]
    
    # Gán màu theo mức nguy hiểm
    colors = []
    for label in sorted_labels:
        info = get_disease_risk_info(label)
        colors.append(info['risk_color'])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_labels,
        x=sorted_values,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0),
            opacity=0.85,
        ),
        text=[f"{v:.1f}%" for v in sorted_values],
        textposition='outside',
        textfont=dict(color='#94a3b8', size=11),
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Xác suất (%)",
            titlefont=dict(color='#64748b'),
            tickfont=dict(color='#64748b'),
            gridcolor='rgba(100, 116, 139, 0.15)',
            range=[0, max(sorted_values) * 1.3]
        ),
        yaxis=dict(
            tickfont=dict(color='#94a3b8', size=11),
            autorange='reversed',
        ),
        margin=dict(l=280, r=60, t=20, b=40),
        height=700,
    )
    
    return fig


# ============================================================
# HÀM PHÁT HIỆN VÙNG DA (SKIN DETECTION)
# ============================================================
def detect_skin_region(img_bgr):
    """Phát hiện vùng da trên ảnh bằng HSV Color Masking.
    Trả về bounding box (x, y, w, h) của vùng da lớn nhất, hoặc None."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Dải màu da người trong HSV
    lower_skin1 = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin1 = np.array([20, 150, 255], dtype=np.uint8)
    lower_skin2 = np.array([160, 30, 60], dtype=np.uint8)
    upper_skin2 = np.array([180, 150, 255], dtype=np.uint8)
    
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    mask = mask1 | mask2
    
    # Làm mịn mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Lấy contour lớn nhất
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    
    # Bỏ qua nếu vùng da quá nhỏ (< 5% diện tích ảnh)
    h, w = img_bgr.shape[:2]
    if area < 0.05 * h * w:
        return None
    
    return cv2.boundingRect(largest)


# ============================================================
# BIẾN CHIA SẺ GIỮA THREAD VIDEO VÀ MAIN THREAD
# ============================================================
realtime_lock = threading.Lock()
realtime_result = {
    "last_analysis_time": 0,
    "top3_names": [],
    "top3_probs": [],
    "bbox": None,
    "analyzing": False
}


def realtime_video_callback(frame):
    """Callback xử lý mỗi frame video từ WebRTC."""
    global realtime_result
    
    img = frame.to_ndarray(format="bgr24")
    h_frame, w_frame = img.shape[:2]
    current_time = time.time()
    
    # 1. Phát hiện vùng da
    bbox = detect_skin_region(img)
    
    # 2. Vẽ bounding box xung quanh vùng da
    if bbox is not None:
        x, y, w, h = bbox
        # Vẽ khung viền neon xanh lá
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 128), 3)
        
        # Vẽ 4 góc nhấn mạnh
        corner_len = min(w, h) // 5
        color_corner = (0, 255, 200)
        # Top-left
        cv2.line(img, (x, y), (x + corner_len, y), color_corner, 5)
        cv2.line(img, (x, y), (x, y + corner_len), color_corner, 5)
        # Top-right
        cv2.line(img, (x + w, y), (x + w - corner_len, y), color_corner, 5)
        cv2.line(img, (x + w, y), (x + w, y + corner_len), color_corner, 5)
        # Bottom-left
        cv2.line(img, (x, y + h), (x + corner_len, y + h), color_corner, 5)
        cv2.line(img, (x, y + h), (x, y + h - corner_len), color_corner, 5)
        # Bottom-right
        cv2.line(img, (x + w, y + h), (x + w - corner_len, y + h), color_corner, 5)
        cv2.line(img, (x + w, y + h), (x + w, y + h - corner_len), color_corner, 5)
        
        # Label "SCANNING" trên khung
        cv2.putText(img, "SCANNING...", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
    
    # 3. Chạy AI mỗi 2 giây
    if model3 is not None and bbox is not None and (current_time - realtime_result["last_analysis_time"]) > 2.0:
        with realtime_lock:
            realtime_result["analyzing"] = True
        
        try:
            x, y, w, h = bbox
            # Cắt vùng da, thêm padding 10%
            pad = int(max(w, h) * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_frame, x + w + pad)
            y2 = min(h_frame, y + h + pad)
            crop = img[y1:y2, x1:x2]
            
            if crop.size > 0:
                # Chuyển sang RGB và chạy qua model
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                val_transforms = get_val_transforms()
                # TTA: Test-Time Augmentation (3 variations)
                # 1. Gốc
                tensor_img1 = val_transforms(image=crop_rgb)["image"].unsqueeze(0)
                # 2. Lật ngang
                crop_flipped = cv2.flip(crop_rgb, 1)
                tensor_img2 = val_transforms(image=crop_flipped)["image"].unsqueeze(0)
                # 3. Tăng sáng nhẹ
                crop_bright = cv2.convertScaleAbs(crop_rgb, alpha=1.1, beta=15)
                tensor_img3 = val_transforms(image=crop_bright)["image"].unsqueeze(0)
                
                # Nạp cả 3 ảnh vào GPU cùng lúc
                tensor_imgs = torch.cat([tensor_img1, tensor_img2, tensor_img3], dim=0).to(device)
                
                # Tạo clinical features mặc định và copy ra 3 bản
                default_clinical = torch.tensor([[0.45, 0.5, 0.2, 0.08, 0.0, 0.4, 0.0]], dtype=torch.float32).to(device)
                batch_clinical = default_clinical.repeat(3, 1)
                
                with torch.no_grad():
                    logits = model3(tensor_imgs, batch_clinical)
                    # Tính softmax cho 3 bản, rồi lấy xác suất trung bình
                    probs = torch.softmax(logits, dim=1).mean(dim=0).cpu()
                
                top3_prob, top3_idx = torch.topk(probs, 3)
                
                with realtime_lock:
                    realtime_result["top3_names"] = [CLASSES[top3_idx[i]] for i in range(3)]
                    realtime_result["top3_probs"] = [top3_prob[i].item() for i in range(3)]
                    realtime_result["bbox"] = bbox
                    realtime_result["last_analysis_time"] = current_time
                    realtime_result["analyzing"] = False
        except Exception:
            with realtime_lock:
                realtime_result["analyzing"] = False
    
    # 4. Hiển thị kết quả lên frame
    with realtime_lock:
        names = realtime_result["top3_names"]
        probs_display = realtime_result["top3_probs"]
    
    if names:
        # Vẽ panel kết quả ở góc trên trái
        panel_h = 130
        panel_w = min(w_frame - 20, 520)
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (15, 23, 42), -1)
        img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
        cv2.rectangle(img, (10, 10), (10 + panel_w, 10 + panel_h), (0, 255, 128), 2)
        
        # Header
        cv2.putText(img, "DermaSense AI - REALTIME", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 118), 2)
        
        # Top 3 kết quả
        colors = [(0, 200, 255), (0, 180, 200), (0, 150, 170)]  # Vàng, Cyan, Teal
        for i, (name, prob) in enumerate(zip(names, probs_display)):
            short_name = name[:35] + "..." if len(name) > 35 else name
            y_pos = 60 + i * 28
            # Thanh progress bar
            bar_w = int(prob * (panel_w - 30))
            cv2.rectangle(img, (20, y_pos - 2), (20 + bar_w, y_pos + 16), colors[i], -1)
            cv2.rectangle(img, (20, y_pos - 2), (10 + panel_w - 10, y_pos + 16), (100, 100, 100), 1)
            # Text
            label = f"{i+1}. {short_name}: {prob:.1%}"
            cv2.putText(img, label, (25, y_pos + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================================
# KHU VỰC CHÍNH
# ============================================================
tab_upload, tab_camera, tab_realtime = st.tabs(["📤 Tải ảnh lên", "📸 Chụp ảnh", "🎥 Phân tích Trực tiếp"])

with tab_upload:
    uploaded_file = st.file_uploader("Chọn ảnh da liễu từ máy tính của bạn", type=["jpg", "jpeg", "png"])
    
with tab_camera:
    camera_file = st.camera_input("Sử dụng webcam để chụp ảnh da liễu")

with tab_realtime:
    st.markdown("""
    <div class="glass-card" style="padding:16px 20px; margin-bottom:16px;">
        <p style="color:#00e676; font-weight:700; margin:0 0 6px 0;">🎥 Chế độ Phân tích Trực tiếp (Real-time)</p>
        <p style="color:#94a3b8; font-size:0.85rem; margin:0;">
            Đưa vùng da cần kiểm tra trước camera. AI sẽ tự động phát hiện vùng da, 
            vẽ khung viền xung quanh và hiển thị kết quả chẩn đoán Top 3 bệnh mỗi 2 giây.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if model3 is None:
        st.warning("⚠️ Chưa có trọng số mô hình. Vui lòng huấn luyện trước khi sử dụng chế độ trực tiếp.")
    else:
        webrtc_ctx = webrtc_streamer(
            key="dermasense-realtime",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=realtime_video_callback,
            media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
            async_processing=True,
        )
        
        st.markdown("""
        <div style="text-align:center; padding:8px;">
            <p style="color:#475569; font-size:0.75rem;">💡 Mẹo: Giữ vùng da cách camera khoảng 15-30cm để AI phân tích tốt nhất</p>
        </div>
        """, unsafe_allow_html=True)

# Ưu tiên ảnh chụp từ camera nếu có, nếu không thì dùng ảnh tải lên
input_file = camera_file if camera_file is not None else uploaded_file

if input_file is not None:
    file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    col_img, col_btn = st.columns([3, 1])
    with col_img:
        st.image(img_rgb, caption="Ảnh tải lên", use_container_width=True)
    with col_btn:
        st.markdown("<br><br>", unsafe_allow_html=True)
        run_analysis = st.button("🚀 Chạy Phân Tích AI", type="primary", use_container_width=True)
    
    if run_analysis:
        # Xử lý các giá trị None (nếu người dùng không nhập thì dùng giá trị trung bình/mặc định)
        final_age = age if age is not None else 45
        final_sex = sex if sex is not None else "male"
        final_site = site if site is not None else "anterior torso"
        final_duration = duration if duration is not None else 30
        final_symptom = symptom if symptom is not None else "none"
        final_skin_type = skin_type if skin_type is not None else "III"
        final_family = family_history if family_history is not None else "no"
        
        # Encode metadata
        age_encoded = final_age / 100.0
        sex_encoded = 1.0 if final_sex == "male" else 0.0
        site_dict_map = {"head/neck": 0.0, "anterior torso": 0.2, "posterior torso": 0.4, "upper extremity": 0.6, "lower extremity": 0.8, "palms/soles": 1.0}
        symp_dict_map = {"none": 0.0, "itch": 0.3, "pain": 0.6, "bleeding": 1.0}
        skin_type_dict_map = {"I": 0.0, "II": 0.2, "III": 0.4, "IV": 0.6, "V": 0.8, "VI": 1.0}
        fh_encoded = 1.0 if final_family == "yes" else 0.0
        
        clinical_features = torch.tensor([[age_encoded, sex_encoded, site_dict_map[final_site], final_duration/365.0, symp_dict_map[final_symptom], skin_type_dict_map[final_skin_type], fh_encoded]], dtype=torch.float32)

        with st.spinner("⏳ Hệ thống đang phân tích toàn diện..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                cv2.imwrite(tmp.name, img_bgr)
                tmp_path = tmp.name
            
            try:
                    # ==========================================
                    # BƯỚC 1: TIỀN XỬ LÝ (STAGE 1)
                    # ==========================================
                    processed_img, is_blurry, is_skin = preprocess_single_image(tmp_path, stage1_config["preprocessing"])
                
                    st.markdown("""
                    <div class="step-indicator fade-in">
                        <div class="step-number">1</div>
                        <div>
                            <div class="step-title">Tiền Xử Lý Ảnh</div>
                            <div class="step-desc">Lọc nhiễu, cân bằng màu, kiểm tra chất lượng</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                    if not is_skin:
                        st.error("🚨 **LOẠI BỎ:** Hình ảnh không chứa vùng da người. Vui lòng tải lên ảnh da liễu.")
                    elif is_blurry and not stage1_config["preprocessing"]["blur_detection"]["flag_only"]:
                        st.warning("🚨 **LOẠI BỎ:** Ảnh quá mờ, không đạt tiêu chuẩn y khoa. Vui lòng chụp lại.")
                    else:
                        st.success("✅ Ảnh đạt chuẩn y khoa — Đã lọc nhiễu và cân bằng màu thành công.")
                    
                        if processed_img is not None:
                            processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                            val_transforms = get_val_transforms()
                            tensor_img = val_transforms(image=processed_rgb)["image"].unsqueeze(0).to(device)
                        
                            # ==========================================
                            # BƯỚC 3: CHẨN ĐOÁN ĐA PHƯƠNG THỨC (STAGE 3)
                            # ==========================================
                            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                            st.markdown("""
                            <div class="step-indicator fade-in fade-in-delay-2">
                                <div class="step-number">3</div>
                                <div>
                                    <div class="step-title">Chẩn Đoán Đa Phương Thức</div>
                                    <div class="step-desc">Kết hợp Hình ảnh + Bệnh án lâm sàng → Phân loại 23 bệnh</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                            if model3 is not None:
                                with torch.no_grad():
                                    logits3 = model3(tensor_img, clinical_features.to(device))
                                    probs3 = torch.softmax(logits3, dim=1)[0]
                            
                                # Tính confidence metrics
                                conf_metrics = compute_confidence_metrics(probs3)
                            
                                # ===== TABS =====
                                tab1, tab2, tab3 = st.tabs(["🔬 Kết Quả Chẩn Đoán", "🧠 Giải Thích AI (XAI)", "📊 Phân Tích Chi Tiết"])
                            
                                with tab1:
                                    # ----- CONFIDENCE METER -----
                                    stars_display = format_star_display(conf_metrics['star_rating'])
                                    st.markdown(f"""
                                    <div class="confidence-meter fade-in">
                                        <p style="color:#64748b; font-size:0.85rem; margin:0;">Độ Tin Cậy Của AI</p>
                                        <div class="confidence-stars">{stars_display}</div>
                                        <p class="confidence-label" style="color:{conf_metrics['confidence_color']};">
                                            {conf_metrics['confidence_level']}
                                        </p>
                                        <p class="confidence-score">
                                            Điểm tổng hợp: {conf_metrics['confidence_score']:.0%} · 
                                            Entropy: {conf_metrics['normalized_entropy']:.2f} · 
                                            Margin: {conf_metrics['margin']:.1%}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                    st.markdown("<br>", unsafe_allow_html=True)
                                
                                    # ----- TOP 3 KẾT QUẢ -----
                                    top3_prob, top3_idx = torch.topk(probs3, 3)
                                
                                    for rank, i in enumerate(range(3)):
                                        cls_name = CLASSES[top3_idx[i]]
                                        p = top3_prob[i].item()
                                        risk_info = get_disease_risk_info(cls_name)
                                    
                                        # Xác định class CSS
                                        if risk_info['risk_color'] == "#ff1744":
                                            card_class = "danger"
                                        elif risk_info['risk_color'] == "#ff9800":
                                            card_class = "warning"
                                        else:
                                            card_class = "safe"
                                    
                                        # Badge urgency
                                        urgency = risk_info['urgency']
                                        badge_class = "urgent" if "KHẨN" in urgency else ("warning" if "NÊN" in urgency else "safe")
                                    
                                        st.markdown(f"""
                                        <div class="result-card {card_class} fade-in fade-in-delay-{rank+1}">
                                            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                                                <div style="flex:1;">
                                                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                                                        <span style="font-size:1.3rem;">{risk_info['icon']}</span>
                                                        <span class="result-name">{cls_name}</span>
                                                        <span class="urgency-badge {badge_class}">{urgency}</span>
                                                    </div>
                                                    <div class="progress-container">
                                                        <div class="progress-bar {card_class}" style="width:{p*100:.1f}%;"></div>
                                                    </div>
                                                </div>
                                                <div style="text-align:right; min-width:80px;">
                                                    <p class="result-prob {card_class}">{p:.1%}</p>
                                                    <p style="color:#64748b; font-size:0.75rem; margin:0;">#{rank+1}</p>
                                                </div>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                    # ----- KHUYẾN NGHỊ Y KHOA -----
                                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                                
                                    top1_name = CLASSES[top3_idx[0]]
                                    top1_risk = get_disease_risk_info(top1_name)
                                    card_class = "danger" if top1_risk['risk_color'] == "#ff1744" else ("warning" if top1_risk['risk_color'] == "#ff9800" else "safe")
                                
                                    st.markdown(f"""
                                    <div class="recommendation-box {card_class}">
                                        <div class="recommendation-title">💊 Khuyến Nghị Y Khoa — {top1_name}</div>
                                        <div class="recommendation-text">{top1_risk['recommendation']}</div>
                                        <br>
                                        <p style="color:#475569; font-size:0.78rem; font-style:italic; margin:0;">
                                            ⚕️ Lưu ý: Kết quả AI chỉ mang tính chất tham khảo, không thay thế chẩn đoán y khoa chính thức. 
                                            Vui lòng tham vấn bác sĩ chuyên khoa Da liễu trước khi quyết định phương án điều trị.
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                                with tab2:
                                    # ===== XAI: GRAD-CAM =====
                                    st.markdown("""
                                    <div class="glass-card">
                                        <h3>🧠 Bản Đồ Nhiệt Grad-CAM — AI Đang Nhìn Vào Đâu?</h3>
                                        <p style="color:#94a3b8; font-size:0.9rem;">
                                            Grad-CAM (Gradient-weighted Class Activation Mapping) tạo bản đồ nhiệt cho thấy 
                                            vùng nào trên ảnh da đóng vai trò quan trọng nhất trong quyết định của AI.
                                            <br><span style="color:#38bdf8;">Vùng đỏ/vàng</span> = AI tập trung cao · 
                                            <span style="color:#64748b;">Vùng xanh/đen</span> = AI ít quan tâm
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                    if gradcam_engine is not None:
                                        try:
                                            # Tạo tensor mới cho Grad-CAM (cần gradient)
                                            tensor_for_cam = val_transforms(image=processed_rgb)["image"].unsqueeze(0).to(device)
                                            clinical_for_cam = clinical_features.clone().to(device)
                                        
                                            # Tạo overlay
                                            overlay, heatmap_colored, intensity_score = gradcam_engine.generate_overlay(
                                                original_image=processed_rgb,
                                                image_tensor=tensor_for_cam,
                                                clinical_tensor=clinical_for_cam,
                                                target_class=top3_idx[0].item(),
                                                alpha=0.5
                                            )
                                        
                                            # Hiển thị 3 cột: Ảnh gốc | Heatmap | Overlay
                                            col_orig, col_heat, col_over = st.columns(3)
                                            with col_orig:
                                                st.image(processed_rgb, caption="Ảnh gốc (Sau tiền xử lý)", use_container_width=True)
                                            with col_heat:
                                                st.image(heatmap_colored, caption="Bản đồ nhiệt Grad-CAM", use_container_width=True)
                                            with col_over:
                                                st.image(overlay, caption="Overlay (Ảnh + Heatmap)", use_container_width=True)
                                        
                                            # Metrics XAI
                                            st.markdown("<br>", unsafe_allow_html=True)
                                            m1, m2, m3 = st.columns(3)
                                            with m1:
                                                st.markdown(f"""
                                                <div class="xai-metric">
                                                    <div class="xai-metric-value">{intensity_score:.0%}</div>
                                                    <div class="xai-metric-label">Độ Tập Trung (Focality)</div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                            with m2:
                                                st.markdown(f"""
                                                <div class="xai-metric">
                                                    <div class="xai-metric-value">{CLASSES[top3_idx[0]][:20]}...</div>
                                                    <div class="xai-metric-label">Bệnh Được Phân Tích</div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                            with m3:
                                                st.markdown(f"""
                                                <div class="xai-metric">
                                                    <div class="xai-metric-value">{top3_prob[0].item():.1%}</div>
                                                    <div class="xai-metric-label">Xác Suất Dự Đoán</div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                        
                                            # Giải thích Grad-CAM
                                            st.markdown(f"""
                                            <div class="glass-card" style="margin-top:16px;">
                                                <h3>📖 Giải Thích Kết Quả</h3>
                                                <p style="color:#cbd5e1; line-height:1.8;">
                                                    {'🔴 <b>Độ tập trung CAO</b>: AI nhìn vào một vùng rất cụ thể trên ảnh. Đây thường là dấu hiệu tốt — AI đã phát hiện ra đặc điểm bệnh lý rõ ràng.' if intensity_score > 0.6 else '🟡 <b>Độ tập trung TRUNG BÌNH</b>: AI phân tích trên nhiều vùng da khác nhau. Có thể tổn thương phân bố rải rác hoặc AI cần thêm thông tin để chắc chắn hơn.' if intensity_score > 0.3 else '🔵 <b>Độ tập trung THẤP</b>: AI nhìn tổng quát toàn bộ ảnh, chưa tập trung vào một điểm cụ thể. Bạn nên cung cấp ảnh chụp cận cảnh hơn vào vùng tổn thương.'}
                                                </p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        except Exception as e:
                                            st.warning(f"⚠️ Không thể tạo Grad-CAM: {str(e)}")
                                    else:
                                        st.info("ℹ️ Module Grad-CAM chưa được khởi tạo. Cần có model Stage 3 để sử dụng tính năng này.")
                            
                                with tab3:
                                    # ===== BIỂU ĐỒ PHÂN TÍCH =====
                                    st.markdown("""
                                    <div class="glass-card">
                                        <h3>📊 Phân Bố Xác Suất — Tất Cả 23 Bệnh Da Liễu</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                    chart_type = st.radio(
                                        "Chọn loại biểu đồ:",
                                        ["🕸️ Radar Chart (Top 8)", "📊 Bar Chart (23 bệnh)"],
                                        horizontal=True
                                    )
                                
                                    if "Radar" in chart_type:
                                        fig = create_radar_chart(probs3, CLASSES)
                                    else:
                                        fig = create_bar_chart(probs3, CLASSES)
                                
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                    # Bảng dữ liệu chi tiết
                                    st.markdown("""
                                    <div class="glass-card">
                                        <h3>📋 Bảng Dữ Liệu Chi Tiết</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                    # XAI Metrics tổng quan
                                    mc1, mc2, mc3, mc4 = st.columns(4)
                                    with mc1:
                                        st.markdown(f"""
                                        <div class="xai-metric">
                                            <div class="xai-metric-value">{conf_metrics['top1_prob']:.1%}</div>
                                            <div class="xai-metric-label">Top-1 Probability</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    with mc2:
                                        st.markdown(f"""
                                        <div class="xai-metric">
                                            <div class="xai-metric-value">{conf_metrics['margin']:.1%}</div>
                                            <div class="xai-metric-label">Margin (Top1 - Top2)</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    with mc3:
                                        st.markdown(f"""
                                        <div class="xai-metric">
                                            <div class="xai-metric-value">{conf_metrics['normalized_entropy']:.3f}</div>
                                            <div class="xai-metric-label">Normalized Entropy</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    with mc4:
                                        st.markdown(f"""
                                        <div class="xai-metric">
                                            <div class="xai-metric-value" style="color:{conf_metrics['confidence_color']};">{conf_metrics['confidence_level']}</div>
                                            <div class="xai-metric-label">Mức Tin Cậy</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                    st.markdown("<br>", unsafe_allow_html=True)
                                
                                    # Bảng toàn bộ 23 bệnh
                                    import pandas as pd
                                    table_data = []
                                    for i in range(len(CLASSES)):
                                        risk_info = get_disease_risk_info(CLASSES[i])
                                        table_data.append({
                                            "Bệnh": f"{risk_info['icon']} {CLASSES[i]}",
                                            "Xác suất": f"{probs3[i].item():.2%}",
                                            "Mức nguy hiểm": risk_info['risk_level'],
                                            "Khẩn cấp": risk_info['urgency'],
                                        })
                                
                                    df = pd.DataFrame(table_data)
                                    df = df.sort_values("Xác suất", ascending=False).reset_index(drop=True)
                                    df.index = df.index + 1
                                    st.dataframe(df, use_container_width=True, height=500)
                        
                            else:
                                st.warning("⚠️ Chưa có trọng số cho Stage 3 (Multimodal). Vui lòng huấn luyện mô hình trước.")
                            
            finally:
                os.remove(tmp_path)
else:
    # Placeholder khi chưa upload ảnh
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:60px 40px;">
        <p style="font-size:3rem; margin-bottom:16px;">📸</p>
        <h3 style="color:#e2e8f0; margin-bottom:8px;">Tải lên ảnh da liễu để bắt đầu</h3>
        <p style="color:#64748b; max-width:500px; margin:0 auto;">
            Hệ thống sẽ phân tích ảnh qua 3 giai đoạn: Tiền xử lý → Sàng lọc nhanh → Chẩn đoán đa phương thức 23 bệnh, 
            kèm theo bản đồ nhiệt Grad-CAM giải thích quyết định của AI.
        </p>
    </div>
    """, unsafe_allow_html=True)
