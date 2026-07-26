import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random

def create_dummy_data():
    raw_img_dir = Path("data/raw/images")
    raw_img_dir.mkdir(parents=True, exist_ok=True)
    
    classes = [
        "Acne and Rosacea", "Actinic Keratosis and Skin Cancer", "Atopic Dermatitis", 
        "Bullous Disease", "Cellulitis and Bacterial Infections", "Eczema", 
        "Exanthems and Drug Eruptions", "Hair Loss", "Herpes and HPV", 
        "Light Diseases and Pigmentation", "Lupus and Connective Tissue", 
        "Melanoma and Nevi", "Nail Fungus", "Poison Ivy and Contact Dermatitis", 
        "Psoriasis and Lichen Planus", "Scabies and Lyme", 
        "Seborrheic Keratoses and Benign Tumors", "Systemic Diseases", 
        "Tinea and Fungal Infections", "Urticaria Hives", "Vascular Tumors", 
        "Vasculitis", "Warts and Molluscum"
    ]
    anatom_sites = ["anterior torso", "head/neck", "lower extremity", "upper extremity", "posterior torso"]
    symptoms = ["itch", "bleeding", "pain", "none"]
    skin_types = ["I", "II", "III", "IV", "V", "VI"]
    family_history_opts = ["yes", "no"]
    
    metadata = []
    
    for i in range(50):
        img_id = f"ISIC_{i:07d}"
        
        # Tạo ảnh giả với màu da trung bình BGR (120, 150, 200)
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img[:] = (120, 150, 200)  # BGR cho màu da
        # Thêm nhiễu ngẫu nhiên nhẹ
        noise = np.random.normal(0, 15, (400, 400, 3)).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # Thêm các vệt tối ngẫu nhiên giả làm lông trên một số ảnh
        if random.random() > 0.5:
            for _ in range(5):
                x1, y1 = random.randint(0, 400), random.randint(0, 400)
                x2, y2 = random.randint(0, 400), random.randint(0, 400)
                cv2.line(img, (x1, y1), (x2, y2), (20, 20, 20), random.randint(1, 3))
        
        # Lưu ảnh
        cv2.imwrite(str(raw_img_dir / f"{img_id}.jpg"), img)
        
        # Tạo metadata (tuổi, giới tính, vị trí,...)
        diagnosis = random.choice(classes)
        age = random.randint(20, 85)
        sex = random.choice(["male", "female"])
        anatom_site = random.choice(anatom_sites)
        duration = random.randint(10, 365)
        symptom = random.choice(symptoms)
        skin_type = random.choice(skin_types)
        family_history = random.choice(family_history_opts)
        
        metadata.append({
            "image_id": img_id,
            "diagnosis": diagnosis,
            "age": age,
            "sex": sex,
            "anatom_site": anatom_site,
            "duration": duration,
            "symptoms": symptom,
            "skin_type": skin_type,
            "family_history": family_history
        })
        
    df = pd.DataFrame(metadata)
    df.to_csv("data/raw/metadata.csv", index=False)
    print("Đã tạo thành công dữ liệu giả: 50 ảnh và metadata.csv (23 lớp)")

if __name__ == "__main__":
    create_dummy_data()
