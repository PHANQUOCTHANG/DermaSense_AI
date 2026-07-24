"""
DermaSense AI — Demo Stage 2: Cảnh báo Nguy cơ (Safety Screener)
"""
import argparse
from pathlib import Path

import cv2
import torch
import yaml
import numpy as np

from src.data.preprocessing import preprocess_single_image
from src.data.augmentations import get_val_transforms
from src.models.vision_branch import VisionBranch

def main():
    parser = argparse.ArgumentParser(description="Demo Hệ thống Sàng lọc (Stage 2)")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn đến ảnh cần dự đoán")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"❌ Không tìm thấy ảnh: {img_path}")
        return

    print(f"\n🔍 Đang xử lý ảnh: {img_path.name}")
    print("-" * 50)

    # 1. Load cấu hình
    with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    with open("configs/stage1_preprocessing.yaml", "r", encoding="utf-8") as f:
        stage1_config = yaml.safe_load(f)
    with open("configs/stage2_train_screener.yaml", "r", encoding="utf-8") as f:
        stage2_config = yaml.safe_load(f)

    # 2. Tiền xử lý (Stage 1)
    print("⏳ Bước 1: Tiền xử lý (Dọn dẹp ảnh)...")
    try:
        processed_img, is_blurry, is_skin = preprocess_single_image(str(img_path), stage1_config["preprocessing"])
    except Exception as e:
        print(f"❌ Lỗi tiền xử lý: {e}")
        return

    if not is_skin:
        print("❌ HỆ THỐNG TỪ CHỐI: Ảnh này không giống ảnh chụp da.")
        return

    if is_blurry and not stage1_config["preprocessing"]["blur_detection"]["flag_only"]:
        print("❌ HỆ THỐNG TỪ CHỐI: Ảnh quá mờ, không thể chẩn đoán chính xác.")
        return

    print("✅ Tiền xử lý hoàn tất (Sạch lông, cân bằng màu, chuẩn kích thước).")

    # Lưu lại ảnh đã xử lý để xem
    out_dir = Path("outputs/demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"processed_{img_path.name}"
    cv2.imwrite(str(out_path), processed_img)

    # 3. Chạy qua mô hình (Stage 2)
    print("\n⏳ Bước 2: AI đang phân tích nguy cơ...")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionBranch(
        model_name=stage2_config["model"]["name"],
        num_classes=2,
        pretrained=False
    ).to(device)
    
    ckpt_path = Path("models/checkpoints/stage_a/best_model.pt")
    if not ckpt_path.exists():
        print(f"❌ Không tìm thấy trọng số mô hình tại {ckpt_path}.")
        print("💡 Gợi ý: Chạy file create_dummy_checkpoint.py hoặc tự huấn luyện trên cloud trước.")
        return
        
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Chuẩn bị ảnh đầu vào
    # Chuyển BGR (OpenCV) sang RGB
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    
    # Áp dụng transforms của Validation
    val_transforms = get_val_transforms()
    tensor_img = val_transforms(image=rgb_img)["image"].unsqueeze(0).to(device)

    # Dự đoán
    with torch.no_grad():
        logits = model(tensor_img)
        probs = torch.softmax(logits, dim=1)[0]
        
    prob_low = probs[0].item()
    prob_high = probs[1].item()
    
    print("\n" + "=" * 50)
    print("📊 KẾT QUẢ DỰ ĐOÁN:")
    print("=" * 50)
    print(f"- Xác suất LÀNH TÍNH (Low Risk): {prob_low:.2%}")
    print(f"- Xác suất CẦN CHÚ Ý (High Risk): {prob_high:.2%}")
    print("-" * 50)
    
    if prob_high > 0.5:
        print("⚠️ CẢNH BÁO: Phát hiện dấu hiệu rủi ro cao (Khả năng có thể là Melanoma, BCC, hoặc AK).")
        print("🩺 KHUYẾN NGHỊ: Người dùng nên đi khám bác sĩ da liễu ngay lập tức!")
    else:
        print("✅ KẾT LUẬN: Tổn thương da có vẻ lành tính (Nevus, VASC...).")
        print("🩺 KHUYẾN NGHỊ: Tiếp tục theo dõi định kỳ, không có gì đáng lo ngại.")
        
    print("=" * 50)
    print(f"📂 Ảnh sau tiền xử lý đã được lưu tại: {out_path}")

if __name__ == "__main__":
    main()
