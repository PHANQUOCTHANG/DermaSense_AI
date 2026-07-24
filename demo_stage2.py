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

    print(f"\n[INFO] Dang xu ly anh: {img_path.name}")
    print("-" * 50)

    # 1. Load cấu hình
    with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    with open("configs/stage1_preprocessing.yaml", "r", encoding="utf-8") as f:
        stage1_config = yaml.safe_load(f)
    with open("configs/stage2_train_screener.yaml", "r", encoding="utf-8") as f:
        stage2_config = yaml.safe_load(f)

    # 2. Tiền xử lý (Stage 1)
    print("... Buoc 1: Tien xu ly (Don dep anh)...")
    try:
        processed_img, is_blurry, is_skin = preprocess_single_image(str(img_path), stage1_config["preprocessing"])
    except Exception as e:
        print(f"❌ Lỗi tiền xử lý: {e}")
        return

    if not is_skin:
        print("[-] HE THONG TU CHOI: Anh nay khong giong anh chup da.")
        return

    if is_blurry and not stage1_config["preprocessing"]["blur_detection"]["flag_only"]:
        print("[-] HE THONG TU CHOI: Anh qua mo, khong the chan doan chinh xac.")
        return

    print("[+] Tien xu ly hoan tat (Sach long, can bang mau, chuan kich thuoc).")

    # Lưu lại ảnh đã xử lý để xem
    out_dir = Path("outputs/demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"processed_{img_path.name}"
    cv2.imwrite(str(out_path), processed_img)

    # 3. Chạy qua mô hình (Stage 2)
    print("\n... Buoc 2: AI dang phan tich nguy co...")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionBranch(
        model_name=stage2_config["model"]["name"],
        num_classes=2,
        pretrained=False
    ).to(device)
    
    ckpt_path = Path("models/checkpoints/stage_a/best_model.pt")
    if not ckpt_path.exists():
        print(f"[-] Khong tim thay trong so mo hinh tai {ckpt_path}.")
        print("[!] Goi y: Chay file create_dummy_checkpoint.py hoac tu huan luyen tren cloud truoc.")
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
    print("KET QUA DU DOAN:")
    print("=" * 50)
    print(f"- Xac suat LANH TINH (Low Risk): {prob_low:.2%}")
    print(f"- Xac suat CAN CHU Y (High Risk): {prob_high:.2%}")
    print("-" * 50)
    
    if prob_high > 0.5:
        print("[!] CANH BAO: Phat hien dau hieu rui ro cao (Kha nang co the la Melanoma, BCC, hoac AK).")
        print("[!] KHUYEN NGHI: Nguoi dung nen di kham bac si da lieu ngay lap tuc!")
    else:
        print("[+] KET LUAN: Ton thuong da co ve lanh tinh (Nevus, VASC...).")
        print("[+] KHUYEN NGHI: Tiep tuc theo doi dinh ky, khong co gi dang lo ngai.")
        
    print("=" * 50)
    print(f"[+] Anh sau tien xu ly da duoc luu tai: {out_path}")

if __name__ == "__main__":
    main()
