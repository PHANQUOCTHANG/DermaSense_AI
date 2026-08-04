import os
import sys
from pathlib import Path
import yaml
import torch
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from src.data.preprocessing import preprocess_single_image
from src.data.augmentations import get_val_transforms
from src.models.fusion_net import MultimodalFusionNet
from src.xai.gradcam import GradCAM

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
CORS(app)

# ============================================================
# TU DIEN TIENG VIET CHO 23 BENH
# ============================================================
DISEASE_VN_MAP = {
    "Acne and Rosacea":                          "Mụn trứng cá & Mụn đỏ",
    "Actinic Keratosis and Skin Cancer":          "Dày sừng quang hóa & Ung thư da",
    "Atopic Dermatitis":                          "Viêm da cơ địa",
    "Bullous Disease":                            "Bệnh bọng nước",
    "Cellulitis and Bacterial Infections":        "Viêm mô tế bào & Nhiễm khuẩn",
    "Eczema":                                     "Bệnh chàm",
    "Exanthems and Drug Eruptions":               "Phát ban & Dị ứng thuốc",
    "Hair Loss":                                  "Rụng tóc / Hói",
    "Herpes and HPV":                             "Mụn rộp & HPV",
    "Light Diseases and Pigmentation":            "Rối loạn sắc tố da",
    "Lupus and Connective Tissue":                "Lupus & Bệnh mô liên kết",
    "Melanoma and Nevi":                          "Ung thư hắc tố & Nốt ruồi",
    "Nail Fungus":                                "Nấm móng",
    "Poison Ivy and Contact Dermatitis":          "Viêm da tiếp xúc",
    "Psoriasis and Lichen Planus":                "Vảy nến & Lichen phẳng",
    "Scabies and Lyme":                           "Ghẻ & Bệnh Lyme",
    "Seborrheic Keratoses and Benign Tumors":     "Dày sừng tiết bã & U lành",
    "Systemic Diseases":                          "Bệnh hệ thống toàn thân",
    "Tinea and Fungal Infections":                "Hắc lào & Nấm da",
    "Urticaria Hives":                            "Mề đay & Nổi mẩn",
    "Vascular Tumors":                            "U mạch máu",
    "Vasculitis":                                 "Viêm mạch máu",
    "Warts and Molluscum":                        "Mụn cóc & U mềm lây",
}

# ============================================================
# TU DIEN LOI KHUYEN CHO 23 BENH
# ============================================================
DISEASE_ADVICE = {
    "Acne and Rosacea": {
        "icon": "🟢",
        "level_vn": "Rủi ro Thấp",
        "advice": "Rửa mặt nhẹ nhàng 2 lần/ngày với sữa rửa mặt có pH cân bằng. Tránh nặn mụn. Dùng Benzoyl Peroxide 2.5% hoặc Retinoid bôi theo chỉ định. Ăn ít đồ cay, dầu mỡ và đường.",
        "see_doctor": "Nếu mụn viêm nặng, lan rộng hoặc để lại sẹo, hãy đến khám Da liễu."
    },
    "Actinic Keratosis and Skin Cancer": {
        "icon": "🔴",
        "level_vn": "Rủi ro Cao",
        "advice": "ĐÂY LÀ DẤU HIỆU NGUY HIỂM. Không tự điều trị tại nhà. Cần sinh thiết da để xác định loại tế bào ung thư. Hạn chế tối đa tiếp xúc ánh nắng mặt trời.",
        "see_doctor": "Khẩn cấp đến khám chuyên khoa Da liễu - Ung bướu ngay lập tức."
    },
    "Atopic Dermatitis": {
        "icon": "🟡",
        "level_vn": "Rủi ro Trung bình",
        "advice": "Dưỡng ẩm da ngay sau khi tắm (trong vòng 3 phút). Tránh các dị nguyên như lông thú, bụi, hóa chất. Dùng xà phòng không mùi, không chứa chất tẩy mạnh. Mặc quần áo vải mềm (cotton).",
        "see_doctor": "Nếu bùng phát, dùng Corticosteroid bôi ngắn hạn theo chỉ định bác sĩ."
    },
    "Bullous Disease": {
        "icon": "🔴",
        "level_vn": "Rủi ro Cao",
        "advice": "Tuyệt đối không tự chọc vỡ các bọng nước. Che phủ vùng bọng nước bằng gạc sạch để tránh nhiễm trùng. Đây là bệnh tự miễn nghiêm trọng.",
        "see_doctor": "Nhập viện ngay để làm sinh thiết da và điều trị Corticosteroid/Immunoglobulin liều cao."
    },
    "Cellulitis and Bacterial Infections": {
        "icon": "🟡",
        "level_vn": "Rủi ro Trung bình",
        "advice": "Giữ vùng da bị nhiễm trùng sạch sẽ và khô ráo. Không tự ý nặn mủ hay chọc vào vùng viêm. Kê cao chi bị viêm để giảm sưng.",
        "see_doctor": "Cần kháng sinh (uống hoặc tiêm). Nếu sốt cao, ớn lạnh, vùng đỏ lan nhanh → đến cấp cứu ngay."
    },
    "Eczema": {
        "icon": "🟡",
        "level_vn": "Rủi ro Trung bình",
        "advice": "Dưỡng ẩm thường xuyên bằng kem dưỡng không hương liệu. Tránh tắm nước quá nóng. Cắt ngắn móng tay để tránh gãi trầy xước da. Tránh các chất gây kích ứng như xà phòng mạnh.",
        "see_doctor": "Nếu kéo dài hơn 2 tuần, cần dùng Corticosteroid bôi hoặc thuốc kháng histamine theo chỉ định."
    },
    "Exanthems and Drug Eruptions": {
        "icon": "🟡",
        "level_vn": "Rủi ro Trung bình",
        "advice": "DỪNG NGAY loại thuốc nghi ngờ gây dị ứng. Uống nhiều nước. Có thể dùng thuốc kháng histamine (Loratadine) để giảm ngứa.",
        "see_doctor": "Đến cấp cứu NGAY nếu có phù mặt, môi, lưỡi, khó thở hoặc phát ban toàn thân kèm sốt cao (Hội chứng Stevens-Johnson)."
    },
    "Hair Loss": {
        "icon": "🟢",
        "level_vn": "Rủi ro Thấp",
        "advice": "Tránh stress, ăn đủ protein và kẽm. Tránh buộc tóc quá chặt. Dùng dầu gội nhẹ nhàng không chứa Sulfate. Xoa bóp da đầu để kích thích tuần hoàn máu.",
        "see_doctor": "Nếu rụng tóc từng mảng tròn đột ngột hoặc rụng nhanh trong thời gian ngắn, khám Da liễu để kiểm tra bệnh tự miễn."
    },
    "Herpes and HPV": {
        "icon": "🟡",
        "level_vn": "Rủi ro Trung bình",
        "advice": "Tránh tiếp xúc da kề da với người khác khi đang bùng phát. Rửa tay thường xuyên. Không dùng chung khăn mặt, dao cạo. Tránh stress vì có thể kích thích Herpes tái phát.",
        "see_doctor": "Cần thuốc kháng virus (Acyclovir/Valacyclovir). Phụ nữ cần tầm soát HPV và ung thư cổ tử cung định kỳ."
    },
    "Light Diseases and Pigmentation": {
        "icon": "🟡",
        "level_vn": "Rủi ro Trung bình",
        "advice": "Bôi kem chống nắng SPF 50+ mỗi ngày, kể cả trời râm. Đội mũ và mặc áo chống nắng khi ra ngoài. Tránh tiếp xúc ánh nắng trực tiếp từ 10h-16h.",
        "see_doctor": "Tham khảo Da liễu về kem dưỡng Hydroquinone, Azelaic Acid hoặc liệu pháp ánh sáng IPL/Laser."
    },
    "Lupus and Connective Tissue": {
        "icon": "🔴",
        "level_vn": "Rủi ro Cao",
        "advice": "Tránh hoàn toàn ánh nắng mặt trời (Lupus rất nhạy cảm với UV). Bôi kem chống nắng mọi lúc. Nghỉ ngơi đủ giấc, tránh stress. Không tự ý ngừng thuốc.",
        "see_doctor": "Cần xét nghiệm ANA, Anti-dsDNA và khám Nội khoa - Miễn dịch khẩn cấp. Lupus có thể ảnh hưởng thận, tim, não."
    },
    "Melanoma and Nevi": {
        "icon": "🔴",
        "level_vn": "Rủi ro Cao",
        "advice": "Kiểm tra theo quy tắc ABCDE: (A)symptom - Bất đối xứng, (B)order - Bờ không đều, (C)olor - Màu không đồng đều, (D)iameter - Đường kính >6mm, (E)volution - Thay đổi theo thời gian. Tránh tia UV.",
        "see_doctor": "KHẨN CẤP: Melanoma là ung thư da nguy hiểm nhất. Cần sinh thiết và đánh giá di căn hạch ngay lập tức."
    },
    "Nail Fungus": {
        "icon": "🟢",
        "level_vn": "Rủi ro Thấp",
        "advice": "Giữ móng tay/chân khô ráo và sạch. Cắt móng thường xuyên. Đi tất thoáng khí, thay tất mỗi ngày. Không đi chân trần ở bể bơi, phòng tắm công cộng.",
        "see_doctor": "Bôi thuốc kháng nấm (Ciclopirox) trong 3-6 tháng. Nếu nặng, cần thuốc uống Terbinafine theo chỉ định bác sĩ."
    },
    "Poison Ivy and Contact Dermatitis": {
        "icon": "🟡",
        "level_vn": "Rủi ro Trung bình",
        "advice": "Xác định và tránh xa chất gây dị ứng (hóa chất, cao su, kim loại, thực vật). Rửa ngay vùng da tiếp xúc bằng nước và xà phòng. Dùng kem chứa Calamine để giảm ngứa.",
        "see_doctor": "Dùng Corticosteroid bôi (Hydrocortisone 1%) hoặc uống nếu phản ứng mạnh. Patch test để xác định dị nguyên."
    },
    "Psoriasis and Lichen Planus": {
        "icon": "🟡",
        "level_vn": "Rủi ro Trung bình",
        "advice": "Dưỡng ẩm thường xuyên để tránh da khô và bong vảy. Tránh stress, vì stress là nguyên nhân hàng đầu gây bùng phát vảy nến. Tắm nước ấm (không nóng). Tránh gãi mạnh.",
        "see_doctor": "Liệu pháp ánh sáng UVB, kem bôi Calcipotriol hoặc thuốc sinh học (Biologics) cho ca nặng."
    },
    "Scabies and Lyme": {
        "icon": "🟡",
        "level_vn": "Rủi ro Trung bình",
        "advice": "Ghẻ: Giặt toàn bộ quần áo, ga gối bằng nước nóng >60°C. Tất cả người trong nhà cần điều trị cùng lúc. Lyme: Tìm và gỡ ve bám bằng nhíp (không vặn, không bóp).",
        "see_doctor": "Ghẻ: Bôi Permethrin 5% để qua đêm. Lyme: Kháng sinh Doxycycline 14-21 ngày theo chỉ định bác sĩ."
    },
    "Seborrheic Keratoses and Benign Tumors": {
        "icon": "🟢",
        "level_vn": "Rủi ro Thấp",
        "advice": "Đây thường là u lành tính, không nguy hiểm. Tránh tự ý cắt, bôi thuốc hoặc cào bóc. Theo dõi kích thước và màu sắc theo thời gian.",
        "see_doctor": "Khám Da liễu nếu tổn thương thay đổi nhanh, chảy máu, hoặc kích thước >1cm để loại trừ ung thư."
    },
    "Systemic Diseases": {
        "icon": "🔴",
        "level_vn": "Rủi ro Cao",
        "advice": "Biểu hiện da có thể là dấu hiệu của bệnh nội tạng (gan, thận, tuyến giáp, tiểu đường). Theo dõi thêm các triệu chứng toàn thân: mệt mỏi, vàng da, sụt cân.",
        "see_doctor": "Xét nghiệm máu toàn diện và khám Nội khoa là bắt buộc để tìm nguyên nhân gốc rễ."
    },
    "Tinea and Fungal Infections": {
        "icon": "🟢",
        "level_vn": "Rủi ro Thấp",
        "advice": "Giữ da khô thoáng, đặc biệt các vùng nếp gấp (bẹn, nách, kẽ ngón chân). Thay quần áo, tất hàng ngày. Không dùng chung khăn tắm với người khác.",
        "see_doctor": "Bôi thuốc kháng nấm (Clotrimazole/Ketoconazole) 2-4 tuần. Ca nặng hoặc hắc lào da đầu cần uống Fluconazole."
    },
    "Urticaria Hives": {
        "icon": "🟢",
        "level_vn": "Rủi ro Thấp",
        "advice": "Tránh các tác nhân kích hoạt như thức ăn lạ, nhiệt độ thay đổi đột ngột, stress. Chườm lạnh để giảm ngứa tạm thời. Mặc quần áo rộng rãi, thoáng mát.",
        "see_doctor": "Uống kháng histamine (Loratadine/Cetirizine). Đến cấp cứu NGAY nếu sưng môi/lưỡi/cổ họng hoặc khó thở."
    },
    "Vascular Tumors": {
        "icon": "🔴",
        "level_vn": "Rủi ro Cao",
        "advice": "Không tự ý cào, gãi hoặc tác động lên vùng u. Tránh chấn thương vì có thể gây chảy máu nhiều.",
        "see_doctor": "Cần siêu âm Doppler mạch máu để phân biệt u lành/ác tính. Sinh thiết nếu nghi ngờ Sarcoma Kaposi hoặc Angiosarcoma."
    },
    "Vasculitis": {
        "icon": "🔴",
        "level_vn": "Rủi ro Cao",
        "advice": "Tránh tiếp xúc lạnh, vì lạnh có thể làm co mạch và trầm trọng hơn. Kê cao chân khi nằm. Không đứng yên quá lâu.",
        "see_doctor": "Khẩn cấp: Xét nghiệm CRP, ANCA, sinh thiết da. Điều trị bằng Corticosteroid hoặc ức chế miễn dịch."
    },
    "Warts and Molluscum": {
        "icon": "🟢",
        "level_vn": "Rủi ro Thấp",
        "advice": "Không tự cắt hay cào mụn cóc vì dễ lây lan. Giữ vùng da sạch, khô. Không dùng chung khăn, đồ dùng cá nhân. Tăng cường hệ miễn dịch bằng ăn uống lành mạnh.",
        "see_doctor": "Bôi Acid Salicylic hoặc đốt lạnh (Cryotherapy) tại phòng khám. Thường tự khỏi trong 1-2 năm, đặc biệt ở trẻ em."
    },
}

# ============================================================
# CAU HINH & KHOI TAO MO HINH
# ============================================================
def load_configs():
    with open("configs/stage1_preprocessing.yaml", "r", encoding="utf-8") as f:
        stage1_cfg = yaml.safe_load(f)
    with open("configs/stage3_train_multimodal.yaml", "r", encoding="utf-8") as f:
        stage3_cfg = yaml.safe_load(f)
    with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    return stage1_cfg, stage3_cfg, base_cfg

stage1_config, stage3_config, base_config = load_configs()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading MultimodalFusionNet model...")
model = MultimodalFusionNet(
    vision_model_name=stage3_config["model"]["vision_name"],
    num_classes=stage3_config["model"]["num_classes"],
    clinical_in_features=stage3_config["model"]["clinical_in_features"],
    pretrained=False
).to(device)

ckpt_path = Path("models/checkpoints/stage_b/best_model.pt")
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")
else:
    print("Warning: Model weights not found. Please train the model first.")
    model = None

val_transforms = get_val_transforms(img_size=stage3_config["data"]["img_size"])

# ============================================================
# HAM XU LY ANH CHUNG (File upload & Base64 webcam)
# ============================================================
def decode_image(source):
    """Giai ma anh tu file upload hoac chuoi base64."""
    if 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']
        img_bytes = file.read()
    elif source:
        # Base64 string from webcam
        if ',' in source:
            source = source.split(',')[1]
        img_bytes = base64.b64decode(source)
    else:
        return None
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def run_inference(img_bgr, form_data):
    """Chay toan bo pipeline: preprocessing -> inference -> gradcam."""
    # Luu tam de preprocessing
    temp_path = "temp_upload_flask.jpg"
    cv2.imwrite(temp_path, img_bgr)
    processed_img, is_blurry, is_skin = preprocess_single_image(temp_path, stage1_config["preprocessing"])
    os.remove(temp_path)

    if not is_skin:
        return None, "Anh khong phai anh da nguoi. Vui long thu lai."

    img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    transformed = val_transforms(image=img_rgb)
    img_tensor = transformed["image"].unsqueeze(0).to(device)

    # Clinical features - Enhanced mapping for text inputs
    site_map = {
        "Đầu / Cổ": 0.0, "Ngực / Bụng": 0.2, "Lưng": 0.4, "Tay": 0.6, "Chân": 0.8, "Lòng bàn tay / Chân": 1.0,
        "head/neck": 0.0, "anterior torso": 0.2, "posterior torso": 0.4, "upper extremity": 0.6, "lower extremity": 0.8, "palms/soles": 1.0
    }
    symp_map = {
        "Không có": 0.0, "Ngứa": 0.3, "Đau rát": 0.6, "Chảy máu / Rỉ dịch": 1.0,
        "none": 0.0, "itch": 0.3, "pain": 0.6, "bleeding": 1.0
    }
    skin_map = {"I": 0.0, "II": 0.2, "III": 0.4, "IV": 0.6, "V": 0.8, "VI": 1.0}

    # Custom text handling (fallback rules)
    raw_site = form_data.get("anatom_site", "Ngực / Bụng").strip()
    site_val = 0.2 # default
    for k, v in site_map.items():
        if k.lower() in raw_site.lower():
            site_val = v
            break

    raw_symptom = form_data.get("symptoms", "Không có").strip()
    symptom_val = 0.0 # default
    for k, v in symp_map.items():
        if k.lower() in raw_symptom.lower():
            symptom_val = v
            break

    age      = float(form_data.get("age", 45)) / 100.0
    sex      = 1.0 if form_data.get("sex", "male") == "male" else 0.0
    site     = site_val
    duration = float(form_data.get("duration", 30)) / 365.0
    symptom  = symptom_val
    skin_type = skin_map.get(form_data.get("skin_type", "III"), 0.4)
    fh       = 1.0 if form_data.get("family_history", "no") == "yes" else 0.0

    clinical_tensor = torch.tensor([[age, sex, site, duration, symptom, skin_type, fh]], dtype=torch.float32).to(device)

    # Inference
    with torch.no_grad():
        logits = model(img_tensor, clinical_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]

    probs_np = probs.cpu().numpy()
    classes  = base_config["labels"]["classes"]
    top5_idx = np.argsort(probs_np)[::-1][:5]

    # Format top5 voi ten Tieng Viet
    top5 = []
    for i in top5_idx:
        eng_name = classes[i]
        vn_name  = DISEASE_VN_MAP.get(eng_name, eng_name)
        top5.append({
            "class":     f"{vn_name} ({eng_name})",
            "class_eng": eng_name,
            "prob":      float(probs_np[i])
        })

    primary_eng  = top5[0]["class_eng"]
    primary_prob = top5[0]["prob"]

    # Risk level
    risk_level = "Medium"
    if primary_eng in base_config["risk_levels"]["high"]["classes"]:
        risk_level = "High"
    elif primary_eng in base_config["risk_levels"]["low"]["classes"]:
        risk_level = "Low"

    # Loi khuyen
    advice = DISEASE_ADVICE.get(primary_eng, {
        "icon": "🟡",
        "level_vn": "Can theo doi",
        "advice": "Hay theo doi thay doi cua vung da va den gap bac si da lieu de duoc chan doan chinh xac.",
        "see_doctor": "Kham bac si de co huong dieu tri phu hop."
    })

    # Grad-CAM
    gradcam_engine = GradCAM(model)
    overlay, _, _ = gradcam_engine.generate_overlay(
        original_image=img_rgb,
        image_tensor=img_tensor,
        clinical_tensor=clinical_tensor,
        target_class=int(top5_idx[0])
    )
    # Base64 Processed Image (DullRazor output)
    # processed_img is already in BGR format
    _, buffer = cv2.imencode('.jpg', processed_img)
    processed_base64 = base64.b64encode(buffer).decode('utf-8')

    # Base64 Heatmap
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', overlay_bgr)
    gradcam_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "primary_diagnosis": top5[0]["class"],
        "confidence":        primary_prob,
        "risk_level":        risk_level,
        "top5":              top5,
        "processed_base64":  processed_base64,
        "gradcam_base64":    gradcam_b64,
        "is_blurry":         is_blurry,
        "advice":            advice["advice"],
        "see_doctor":        advice["see_doctor"],
        "risk_icon":         advice["icon"],
        "risk_level_vn":     advice["level_vn"],
    }, None


# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model chua duoc tai (Thieu file best_model.pt)"}), 500
    try:
        b64 = request.form.get("image_b64", "")
        img_bgr = decode_image(b64)
        if img_bgr is None:
            return jsonify({"error": "Khong co anh hop le"}), 400

        result, error = run_inference(img_bgr, request.form)
        if error:
            return jsonify({"error": error}), 400
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Flask Server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
