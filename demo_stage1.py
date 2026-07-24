import argparse
import cv2
import yaml
from pathlib import Path
from src.data.preprocessing import preprocess_single_image

def main():
    parser = argparse.ArgumentParser(description="Demo Tiền xử lý 1 ảnh da liễu (Stage 1)")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn đến ảnh cần kiểm tra")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[X] Không tìm thấy ảnh tại: {image_path}")
        return

    # Tải cấu hình
    with open("configs/stage1_preprocessing.yaml", "r", encoding="utf-8") as f:
        stage1_config = yaml.safe_load(f)

    print(f"\n[*] Đang phân tích ảnh: {image_path.name}")
    print("-" * 40)

    # Chạy pipeline tiền xử lý cho 1 ảnh
    processed_img, is_blurry, is_skin = preprocess_single_image(str(image_path), stage1_config["preprocessing"])

    if not is_skin:
        print("[!] KẾT QUẢ: BỊ LOẠI (REJECTED)")
        print("Lý do: Ảnh KHÔNG PHẢI ẢNH DA (hoặc diện tích da quá nhỏ). Vui lòng tải lên ảnh chụp tổn thương da.")
        return

    if processed_img is None:
        print("[X] Lỗi không thể đọc ảnh.")
        return

    # Đánh giá chất lượng
    if is_blurry:
        print("[!] KẾT QUẢ: BỊ LOẠI (REJECTED)")
        print("Lý do: Ảnh quá mờ, không đủ chi tiết để chẩn đoán y khoa.")
    else:
        print("[OK] KẾT QUẢ: ĐƯỢC CHẤP NHẬN (ACCEPTED)")
        print("Ảnh đạt tiêu chuẩn sắc nét.")
        
        # Lưu kết quả
        out_dir = Path("outputs/demo_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"cleaned_{image_path.name}"
        
        cv2.imwrite(str(out_path), processed_img)
        print("-" * 40)
        print(f"[*] Ảnh đã được tự động làm sạch (xóa lông, chỉnh màu) và lưu tại:")
        print(f"-> {out_path.absolute()}")

if __name__ == "__main__":
    main()
