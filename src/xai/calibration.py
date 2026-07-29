"""
DermaSense AI — Confidence Calibration & Metrics
Module phân tích độ tin cậy của dự đoán AI.

Giúp bác sĩ hiểu:
- AI "tự tin" bao nhiêu phần trăm vào kết quả?
- Khoảng cách giữa dự đoán số 1 và số 2 là bao xa? (margin)
- Entropy của phân phối xác suất (càng thấp = càng chắc chắn).
"""
import numpy as np
from typing import Dict, List, Tuple
import torch


def compute_confidence_metrics(probs: torch.Tensor) -> Dict:
    """Tính toán các chỉ số đo lường độ tin cậy.
    
    Args:
        probs: Tensor xác suất sau softmax, shape (num_classes,).
        
    Returns:
        Dict chứa các metrics:
        - top1_prob: Xác suất cao nhất
        - top2_prob: Xác suất cao thứ 2
        - margin: Chênh lệch giữa top1 và top2
        - entropy: Shannon entropy (đơn vị nats)
        - normalized_entropy: Entropy chuẩn hóa về [0, 1]
        - star_rating: Thang đo 5 sao (1-5)
        - confidence_level: Mức tin cậy text ("Rất cao" → "Rất thấp")
        - confidence_color: Mã màu hex tương ứng
    """
    probs_np = probs.detach().cpu().numpy()
    num_classes = len(probs_np)
    
    # Sắp xếp giảm dần
    sorted_probs = np.sort(probs_np)[::-1]
    
    top1_prob = float(sorted_probs[0])
    top2_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top1_prob - top2_prob
    
    # Shannon Entropy
    # Thêm epsilon nhỏ để tránh log(0)
    eps = 1e-10
    entropy = -np.sum(probs_np * np.log(probs_np + eps))
    max_entropy = np.log(num_classes)  # Entropy tối đa khi phân phối đều
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Tính Confidence Score tổng hợp (kết hợp top1, margin, và entropy)
    # Trọng số: top1 chiếm 40%, margin 30%, (1-entropy) 30%
    confidence_score = (
        0.4 * top1_prob + 
        0.3 * min(margin * 2, 1.0) +  # Scale margin lên vì thường nhỏ
        0.3 * (1 - normalized_entropy)
    )
    confidence_score = np.clip(confidence_score, 0, 1)
    
    # Quy đổi ra 5 sao
    star_rating = _score_to_stars(confidence_score)
    
    # Mức tin cậy và màu sắc
    confidence_level, confidence_color = _score_to_level(confidence_score)
    
    return {
        'top1_prob': top1_prob,
        'top2_prob': top2_prob,
        'margin': margin,
        'entropy': float(entropy),
        'normalized_entropy': float(normalized_entropy),
        'confidence_score': float(confidence_score),
        'star_rating': star_rating,
        'confidence_level': confidence_level,
        'confidence_color': confidence_color,
    }


def _score_to_stars(score: float) -> float:
    """Chuyển confidence score (0-1) thành thang 5 sao."""
    return round(score * 5 * 2) / 2  # Làm tròn 0.5 sao


def _score_to_level(score: float) -> Tuple[str, str]:
    """Chuyển confidence score thành mức tin cậy và mã màu."""
    if score >= 0.8:
        return "Rất cao", "#00e676"      # Xanh lá sáng
    elif score >= 0.6:
        return "Cao", "#4caf50"           # Xanh lá
    elif score >= 0.4:
        return "Trung bình", "#ff9800"    # Cam
    elif score >= 0.2:
        return "Thấp", "#ff5722"          # Đỏ cam
    else:
        return "Rất thấp", "#f44336"      # Đỏ


def format_star_display(star_rating: float) -> str:
    """Tạo chuỗi hiển thị sao đẹp mắt.
    
    Ví dụ: 3.5 → "★★★☆☆" (3 sao đầy + 1 nửa sao + 1 sao trống)
    """
    full_stars = int(star_rating)
    half_star = (star_rating - full_stars) >= 0.5
    empty_stars = 5 - full_stars - (1 if half_star else 0)
    
    display = "★" * full_stars
    if half_star:
        display += "⯪"  # Nửa sao
    display += "☆" * empty_stars
    
    return display


def get_disease_risk_info(class_name: str) -> Dict:
    """Trả về thông tin chi tiết về mức nguy hiểm và khuyến nghị cho từng loại bệnh.
    
    Args:
        class_name: Tên bệnh (tiếng Việt).
        
    Returns:
        Dict chứa: risk_level, risk_color, icon, recommendation, urgency.
    """
    # Bệnh nguy hiểm cao (Đỏ) — Cần khám ngay
    high_risk = {
        "Ung thư da / Dày sừng (AK/Cancer)": {
            "risk_level": "Nguy hiểm cao",
            "risk_color": "#ff1744",
            "icon": "🔴",
            "urgency": "KHẨN CẤP",
            "recommendation": "Cần sinh thiết và khám chuyên khoa Da liễu - Ung bướu NGAY LẬP TỨC. Không tự điều trị tại nhà."
        },
        "Ung thư hắc tố / Nốt ruồi (Melanoma/Nevi)": {
            "risk_level": "Nguy hiểm cao",
            "risk_color": "#ff1744",
            "icon": "🔴",
            "urgency": "KHẨN CẤP",
            "recommendation": "Melanoma là loại ung thư da nguy hiểm nhất. Cần sinh thiết khẩn cấp và theo dõi di căn."
        },
        "Lupus / Bệnh mô liên kết": {
            "risk_level": "Nguy hiểm cao",
            "risk_color": "#ff1744",
            "icon": "🔴",
            "urgency": "KHẨN CẤP",
            "recommendation": "Lupus có thể ảnh hưởng đến nhiều cơ quan nội tạng. Cần xét nghiệm ANA và khám Nội khoa ngay."
        },
        "Bệnh bọng nước (Bullous Disease)": {
            "risk_level": "Nguy hiểm cao",
            "risk_color": "#ff1744",
            "icon": "🔴",
            "urgency": "KHẨN CẤP",
            "recommendation": "Pemphigus/Pemphigoid cần điều trị corticosteroid sớm. Sinh thiết da kèm miễn dịch huỳnh quang."
        },
        "Bệnh hệ thống (Systemic)": {
            "risk_level": "Nguy hiểm cao",
            "risk_color": "#ff1744",
            "icon": "🔴",
            "urgency": "KHẨN CẤP",
            "recommendation": "Biểu hiện da của bệnh toàn thân. Cần xét nghiệm máu toàn diện và khám Nội khoa."
        },
        "U mạch máu (Vascular Tumors)": {
            "risk_level": "Nguy hiểm cao",
            "risk_color": "#ff1744",
            "icon": "🔴",
            "urgency": "CẦN THEO DÕI",
            "recommendation": "Cần phân biệt u lành/ác. Siêu âm Doppler mạch máu và sinh thiết nếu nghi ngờ."
        },
    }
    
    # Bệnh trung bình (Vàng) — Nên khám sớm
    medium_risk = {
        "Viêm da cơ địa (Atopic Dermatitis)": {
            "risk_level": "Trung bình",
            "risk_color": "#ff9800",
            "icon": "🟡",
            "urgency": "NÊN KHÁM",
            "recommendation": "Dưỡng ẩm thường xuyên, tránh dị ứng nguyên. Dùng corticosteroid bôi ngắn hạn nếu bùng phát."
        },
        "Chàm (Eczema)": {
            "risk_level": "Trung bình",
            "risk_color": "#ff9800",
            "icon": "🟡",
            "urgency": "NÊN KHÁM",
            "recommendation": "Giữ ẩm da, tránh xà phòng mạnh. Nếu kéo dài > 2 tuần, dùng thuốc bôi theo chỉ định."
        },
        "Vảy nến / Lichen phẳng (Psoriasis)": {
            "risk_level": "Trung bình",
            "risk_color": "#ff9800",
            "icon": "🟡",
            "urgency": "NÊN KHÁM",
            "recommendation": "Bệnh mãn tính cần quản lý lâu dài. Liệu pháp ánh sáng UVB hoặc thuốc sinh học có thể cần thiết."
        },
        "Viêm mô tế bào / Chốc lở (Bacterial)": {
            "risk_level": "Trung bình",
            "risk_color": "#ff9800",
            "icon": "🟡",
            "urgency": "NÊN KHÁM",
            "recommendation": "Cần kháng sinh đường uống/tiêm. Nếu sốt cao hoặc lan rộng, nhập viện điều trị."
        },
        "Herpes / HPV": {
            "risk_level": "Trung bình",
            "risk_color": "#ff9800",
            "icon": "🟡",
            "urgency": "NÊN KHÁM",
            "recommendation": "Herpes cần thuốc kháng virus (Acyclovir). HPV cần theo dõi và tầm soát ung thư cổ tử cung."
        },
        "Ghẻ / Ve cắn (Scabies/Lyme)": {
            "risk_level": "Trung bình",
            "risk_color": "#ff9800",
            "icon": "🟡",
            "urgency": "NÊN KHÁM",
            "recommendation": "Ghẻ: bôi Permethrin 5%, giặt đồ nóng. Lyme: kháng sinh Doxycycline nếu có ve cắn."
        },
        "Viêm da tiếp xúc (Contact Dermatitis)": {
            "risk_level": "Trung bình",
            "risk_color": "#ff9800",
            "icon": "🟡",
            "urgency": "NÊN KHÁM",
            "recommendation": "Xác định và tránh chất gây dị ứng. Corticosteroid bôi ngắn hạn khi bùng phát."
        },
        "Phát ban dị ứng thuốc (Exanthems)": {
            "risk_level": "Trung bình",
            "risk_color": "#ff9800",
            "icon": "🟡",
            "urgency": "NÊN KHÁM",
            "recommendation": "NGỪNG NGAY thuốc nghi ngờ gây dị ứng. Đến cấp cứu nếu phù mặt/khó thở."
        },
        "Viêm mạch máu (Vasculitis)": {
            "risk_level": "Trung bình",
            "risk_color": "#ff9800",
            "icon": "🟡",
            "urgency": "NÊN KHÁM",
            "recommendation": "Cần xét nghiệm máu (CRP, ANCA). Sinh thiết da để xác định loại viêm mạch."
        },
        "Bệnh sắc tố (Nám/Bạch biến)": {
            "risk_level": "Trung bình",
            "risk_color": "#ff9800",
            "icon": "🟡",
            "urgency": "TÙY CHỌN",
            "recommendation": "Bạch biến: liệu pháp ánh sáng UVB. Nám: kem chống nắng SPF50+ và Hydroquinone theo chỉ định."
        },
    }
    
    # Bệnh nhẹ (Xanh) — Có thể tự theo dõi
    low_risk = {
        "Mụn trứng cá & Mụn đỏ (Acne/Rosacea)": {
            "risk_level": "Nhẹ",
            "risk_color": "#00e676",
            "icon": "🟢",
            "urgency": "TỰ THEO DÕI",
            "recommendation": "Rửa mặt 2 lần/ngày, tránh đồ cay. Dùng Benzoyl Peroxide 2.5% hoặc Retinoid bôi."
        },
        "Rụng tóc (Hair Loss)": {
            "risk_level": "Nhẹ",
            "risk_color": "#00e676",
            "icon": "🟢",
            "urgency": "TỰ THEO DÕI",
            "recommendation": "Minoxidil 5% bôi 2 lần/ngày. Nếu rụng nhiều hoặc từng mảng, khám da liễu."
        },
        "Nấm móng (Nail Fungus)": {
            "risk_level": "Nhẹ",
            "risk_color": "#00e676",
            "icon": "🟢",
            "urgency": "TỰ THEO DÕI",
            "recommendation": "Thuốc bôi kháng nấm (Ciclopirox). Nếu nặng, cần thuốc uống Terbinafine 3-6 tháng."
        },
        "U lành tính (Benign Tumors)": {
            "risk_level": "Nhẹ",
            "risk_color": "#00e676",
            "icon": "🟢",
            "urgency": "TỰ THEO DÕI",
            "recommendation": "Thường không cần điều trị. Theo dõi kích thước, đi khám nếu thay đổi nhanh."
        },
        "Hắc lào / Nấm da (Tinea)": {
            "risk_level": "Nhẹ",
            "risk_color": "#00e676",
            "icon": "🟢",
            "urgency": "TỰ THEO DÕI",
            "recommendation": "Thuốc bôi kháng nấm (Clotrimazole/Ketoconazole) 2-4 tuần. Giữ da khô thoáng."
        },
        "Mề đay (Urticaria Hives)": {
            "risk_level": "Nhẹ",
            "risk_color": "#00e676",
            "icon": "🟢",
            "urgency": "TỰ THEO DÕI",
            "recommendation": "Thuốc kháng histamine (Loratadine/Cetirizine). Đến cấp cứu nếu sưng môi/lưỡi/khó thở."
        },
        "Mụn cóc / U mềm (Warts/Molluscum)": {
            "risk_level": "Nhẹ",
            "risk_color": "#00e676",
            "icon": "🟢",
            "urgency": "TỰ THEO DÕI",
            "recommendation": "Acid Salicylic bôi hoặc đốt lạnh (cryotherapy). Thường tự khỏi trong 1-2 năm."
        },
    }
    
    # Tìm trong các bảng
    all_risks = {**high_risk, **medium_risk, **low_risk}
    
    if class_name in all_risks:
        return all_risks[class_name]
    
    # Mặc định nếu không tìm thấy
    return {
        "risk_level": "Chưa xác định",
        "risk_color": "#9e9e9e",
        "icon": "⚪",
        "urgency": "NÊN KHÁM",
        "recommendation": "Vui lòng đến khám bác sĩ chuyên khoa Da liễu để được tư vấn chi tiết."
    }
