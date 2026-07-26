import os
import glob
import pandas as pd
import random
from pathlib import Path
import shutil

def ingest_dermnet():
    print("Bắt đầu xử lý bộ dữ liệu DermNet...")
    
    # Thư mục gốc chứa DermNet (thường sau khi giải nén từ Kaggle sẽ có các thư mục train, test)
    raw_base = Path("data/raw/images")
    if not raw_base.exists():
        print(f"Lỗi: Không tìm thấy thư mục {raw_base}. Vui lòng tải và giải nén dữ liệu vào đây.")
        return
        
    train_dir = raw_base / "train"
    test_dir = raw_base / "test"
    
    if not train_dir.exists() and not test_dir.exists():
        print("Không tìm thấy cấu trúc thư mục 'train' hoặc 'test' bên trong. Vui lòng kiểm tra lại quá trình giải nén Kaggle.")
        return
        
    # Chuẩn bị metadata giả lập
    anatom_sites = ["anterior torso", "head/neck", "lower extremity", "upper extremity", "posterior torso"]
    symptoms = ["itch", "bleeding", "pain", "none"]
    skin_types = ["I", "II", "III", "IV", "V", "VI"]
    family_history_opts = ["yes", "no"]
    
    metadata = []
    
    # Gom tất cả ảnh từ các thư mục con
    all_images = []
    if train_dir.exists():
        for cls_dir in train_dir.iterdir():
            if cls_dir.is_dir():
                for img_path in cls_dir.glob("*.*"):
                    all_images.append((img_path, cls_dir.name))
                    
    if test_dir.exists():
        for cls_dir in test_dir.iterdir():
            if cls_dir.is_dir():
                for img_path in cls_dir.glob("*.*"):
                    all_images.append((img_path, cls_dir.name))
                    
    print(f"Tìm thấy tổng cộng {len(all_images)} ảnh. Tiến hành chuẩn hóa tên và tạo metadata...")
    
    # Tạo thư mục chứa tất cả ảnh gộp lại
    unified_img_dir = Path("data/raw/unified_images")
    unified_img_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (img_path, original_label) in enumerate(all_images):
        img_id = f"ISIC_{idx:07d}"
        new_path = unified_img_dir / f"{img_id}{img_path.suffix}"
        
        # Di chuyển/Copy ảnh ra thư mục tổng
        shutil.copy2(img_path, new_path)
        
        # Sinh metadata ngẫu nhiên hợp lý
        age = random.randint(15, 80)
        sex = random.choice(["male", "female"])
        anatom_site = random.choice(anatom_sites)
        duration = random.randint(5, 365)
        symptom = random.choice(symptoms)
        skin_type = random.choice(skin_types)
        family_history = random.choice(family_history_opts)
        
        metadata.append({
            "image_id": img_id,
            "diagnosis": original_label, # Lấy tên thư mục làm label
            "age": age,
            "sex": sex,
            "anatom_site": anatom_site,
            "duration": duration,
            "symptoms": symptom,
            "skin_type": skin_type,
            "family_history": family_history
        })
        
        if (idx + 1) % 1000 == 0:
            print(f"  Đã xử lý {idx + 1}/{len(all_images)} ảnh...")
            
    # Lưu metadata.csv
    df = pd.DataFrame(metadata)
    df.to_csv("data/raw/metadata.csv", index=False)
    
    print("\nHoàn tất Ingestion!")
    print(f"- Số lượng ảnh: {len(all_images)}")
    print(f"- Các ảnh đã được copy về thư mục: {unified_img_dir}")
    print(f"- File metadata được lưu tại: data/raw/metadata.csv")
    print("Bạn có thể đổi tên thư mục 'unified_images' thành 'images' (sau khi xóa thư mục 'images' cũ) để huấn luyện.")

if __name__ == "__main__":
    ingest_dermnet()
