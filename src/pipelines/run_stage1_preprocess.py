"""
DermaSense AI — Pipeline Stage 1: Preprocessing
Chạy tiền xử lý trên toàn bộ raw dataset, chia train/val/test và mã hóa clinical metadata.
"""
import os
import yaml
import shutil
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.data.preprocessing import preprocess_single_image

def load_configs():
    with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    with open("configs/stage1_preprocessing.yaml", "r", encoding="utf-8") as f:
        stage1_config = yaml.safe_load(f)
    return base_config, stage1_config

def setup_directories(paths):
    """Xóa (nếu tồn tại) và tạo lại các thư mục đích."""
    processed_dir = Path(paths["data_processed"])
    for split in ["train", "val", "test", "rejected"]:
        split_dir = processed_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        
    clinical_dir = Path(paths["data_clinical"])
    clinical_dir.mkdir(parents=True, exist_ok=True)

def encode_clinical_metadata(df, logger):
    """Mã hóa các trường clinical từ text sang số."""
    logger.info("Encoding clinical metadata...")
    
    encoded_df = df.copy()
    
    # Mã hóa 'age' (StandardScaler)
    if 'age' in encoded_df.columns:
        # Điền các giá trị thiếu bằng giá trị trung vị
        encoded_df['age'] = encoded_df['age'].fillna(encoded_df['age'].median())
        scaler = StandardScaler()
        encoded_df['age'] = scaler.fit_transform(encoded_df[['age']])
        
    # Mã hóa 'sex' (0/1)
    if 'sex' in encoded_df.columns:
        encoded_df['sex'] = encoded_df['sex'].map({'male': 1, 'female': 0}).fillna(0.5)
        
    # Mã hóa 'anatom_site' (LabelEncoder)
    if 'anatom_site' in encoded_df.columns:
        encoded_df['anatom_site'] = encoded_df['anatom_site'].fillna('unknown')
        le = LabelEncoder()
        encoded_df['anatom_site'] = le.fit_transform(encoded_df['anatom_site'])
        
    # Mã hóa 'duration' (StandardScaler)
    if 'duration' in encoded_df.columns:
        encoded_df['duration'] = encoded_df['duration'].fillna(encoded_df['duration'].median())
        scaler = StandardScaler()
        encoded_df['duration'] = scaler.fit_transform(encoded_df[['duration']])
        
    # Mã hóa 'symptoms' (Đơn giản hóa: dùng LabelEncoder thay vì MultiLabel cho phiên bản đầu tiên)
    if 'symptoms' in encoded_df.columns:
        encoded_df['symptoms'] = encoded_df['symptoms'].fillna('none')
        # Cách mã hóa đơn giản để demo
        symptoms_le = LabelEncoder()
        encoded_df['symptoms'] = symptoms_le.fit_transform(encoded_df['symptoms'])
        
    return encoded_df

def main():
    base_config, stage1_config = load_configs()
    paths = base_config["paths"]
    
    logger = get_logger("stage1_preprocess", log_dir=paths["logs"])
    logger.info("="*50)
    logger.info("Start Stage 1: Preprocessing")
    logger.info("="*50)
    
    set_seed(base_config["seed"])
    
    # 1. Khởi tạo và thiết lập các thư mục
    setup_directories(paths)
    
    # 2. Đọc và mã hóa Metadata lâm sàng
    raw_metadata_path = Path(paths["data_raw"]) / "metadata.csv"
    if not raw_metadata_path.exists():
        logger.error(f"Không tìm thấy {raw_metadata_path}")
        return
        
    df = pd.read_csv(raw_metadata_path)
    logger.info(f"Loaded metadata: {len(df)} records")
    
    encoded_df = encode_clinical_metadata(df, logger)
    encoded_metadata_path = Path(paths["data_clinical"]) / "metadata_encoded.csv"
    encoded_df.to_csv(encoded_metadata_path, index=False)
    logger.info(f"Saved encoded metadata to {encoded_metadata_path}")
    
    # 3. Chia tập dữ liệu (Stratified Split)
    train_ratio = base_config["data_split"]["train"]
    val_ratio = base_config["data_split"]["val"]
    test_ratio = base_config["data_split"]["test"]
    
    # Chia phần train và phần còn lại (val+test)
    try:
        train_df, val_test_df = train_test_split(
            df, 
            test_size=(val_ratio + test_ratio), 
            stratify=df['diagnosis'], 
            random_state=base_config["seed"]
        )
    except ValueError:
        logger.warning("Stratified split failed for train/val_test (too few samples). Falling back to random split.")
        train_df, val_test_df = train_test_split(
            df, 
            test_size=(val_ratio + test_ratio), 
            random_state=base_config["seed"]
        )
    
    # Chia phần val và test từ phần còn lại
    test_ratio_relative = test_ratio / (val_ratio + test_ratio)
    try:
        val_df, test_df = train_test_split(
            val_test_df,
            test_size=test_ratio_relative,
            stratify=val_test_df['diagnosis'],
            random_state=base_config["seed"]
        )
    except ValueError:
        logger.warning("Stratified split failed for val/test (too few samples). Falling back to random split.")
        val_df, test_df = train_test_split(
            val_test_df,
            test_size=test_ratio_relative,
            random_state=base_config["seed"]
        )
    
    logger.info(f"Split data: Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)})")
    
    # Tạo ánh xạ từ ID ảnh sang tập dữ liệu (train/val/test)
    split_map = {}
    for idx, row in train_df.iterrows():
        split_map[row['image_id']] = 'train'
    for idx, row in val_df.iterrows():
        split_map[row['image_id']] = 'val'
    for idx, row in test_df.iterrows():
        split_map[row['image_id']] = 'test'
        
    # 4. Tiền xử lý Ảnh
    raw_images_dir = Path(paths["data_raw"]) / "images"
    processed_dir = Path(paths["data_processed"])
    
    image_files = list(raw_images_dir.glob("*.jpg"))
    logger.info(f"Found {len(image_files)} raw images.")
    
    stats = {"processed": 0, "rejected": 0, "errors": 0}
    
    for img_path in tqdm(image_files, desc="Preprocessing images"):
        img_id = img_path.stem
        
        # Kiểm tra nếu img_id không có trong metadata
        if img_id not in split_map:
            # Có thể log warning hoặc skip
            pass
            
        # Lấy tập đích và chẩn đoán bệnh
        if img_id in split_map:
            split = split_map[img_id]
            diagnosis = df[df['image_id'] == img_id]['diagnosis'].values[0]
        else:
            split = "train"
            diagnosis = "UNKNOWN"
            
        # Chạy pipeline tiền xử lý
        try:
            processed_img, is_blurry, is_skin = preprocess_single_image(str(img_path), stage1_config["preprocessing"])
            
            if not is_skin:
                stats["rejected"] += 1
                continue
            
            if processed_img is None:
                stats["errors"] += 1
                continue
                
            if is_blurry and stage1_config["preprocessing"]["blur_detection"]["flag_only"] == False:
                # Lưu vào thư mục bị loại (rejected)
                out_path = processed_dir / "rejected" / img_path.name
                cv2.imwrite(str(out_path), processed_img)
                stats["rejected"] += 1
            else:
                # Tạo thư mục theo tên bệnh bên trong thư mục split
                out_dir = processed_dir / split / diagnosis
                out_dir.mkdir(parents=True, exist_ok=True)
                
                out_path = out_dir / img_path.name
                cv2.imwrite(str(out_path), processed_img)
                stats["processed"] += 1
                
        except Exception as e:
            logger.error(f"Error processing image {img_path.name}: {e}")
            stats["errors"] += 1
            
    logger.info("="*50)
    logger.info("Stage 1 completed!")
    logger.info(f"Stats: Processed: {stats['processed']}, Rejected: {stats['rejected']}, Errors: {stats['errors']}")

if __name__ == "__main__":
    main()
