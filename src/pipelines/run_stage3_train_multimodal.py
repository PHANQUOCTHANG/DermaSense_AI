"""
Pipeline Huấn luyện Stage 3: Mô hình Đa phương thức (Multimodal Fusion)
Áp dụng chiến lược huấn luyện 2 bước (2-Step Training)
"""

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
import numpy as np

from src.data.dataset import DermDataset
from src.data.augmentations import get_train_transforms, get_val_transforms
from src.data.samplers import get_weighted_sampler
from src.models.fusion_net import MultimodalFusionNet
from src.models.loss import MultiClassFocalLoss
from src.utils.seed import set_seed, get_device

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, clinicals, labels in dataloader:
        images, clinicals, labels = images.to(device), clinicals.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images, clinicals)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / total, correct / total

def val_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, clinicals, labels in dataloader:
            images, clinicals, labels = images.to(device), clinicals.to(device), labels.to(device)
            
            logits = model(images, clinicals)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / total, correct / total

def main():
    print("🚀 Bắt đầu Stage 3: Huấn luyện Multimodal (Ảnh + Lâm sàng)")
    
    # 1. Load Cấu hình
    with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    with open("configs/stage3_train_multimodal.yaml", "r", encoding="utf-8") as f:
        stage3_config = yaml.safe_load(f)
        
    set_seed(base_config["seed"])
    device = get_device(base_config["device"])
    
    # 2. Chuẩn bị Dữ liệu
    print("⏳ Đang chuẩn bị dữ liệu...")
    
    # Load toàn bộ dữ liệu từ raw/images vì chưa chạy qua Stage 1 để tiết kiệm thời gian Demo
    full_dataset = DermDataset(
        img_dir='data/raw/images',
        is_binary=False,
        transforms=get_train_transforms(stage3_config.get("augmentations", {})),
        use_clinical=True,
        clinical_csv=stage3_config["data"]["clinical_csv_path"]
    )
    
    # Tự động chia 80% train, 20% val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Chỉnh sửa transforms cho val_dataset
    val_dataset.dataset.transforms = get_val_transforms()
    
    
    # Dummy weight cho các classes để tránh lỗi nếu thư mục không đủ ảnh
    num_classes = stage3_config["model"]["num_classes"]
    class_weights = torch.ones(num_classes).to(device)
    sampler = get_weighted_sampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=stage3_config["data"]["batch_size"],
        sampler=sampler,
        num_workers=stage3_config["data"]["num_workers"],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=stage3_config["data"]["batch_size"],
        shuffle=False,
        num_workers=stage3_config["data"]["num_workers"],
        pin_memory=True
    )
    
    # 3. Khởi tạo Mô hình
    print("🧠 Khởi tạo mô hình MultimodalFusionNet...")
    model = MultimodalFusionNet(
        vision_model_name=stage3_config["model"]["vision_name"],
        num_classes=stage3_config["model"]["num_classes"],
        clinical_in_features=stage3_config["model"]["clinical_in_features"],
        clinical_out_features=stage3_config["model"]["clinical_out_features"],
        pretrained=stage3_config["model"]["pretrained"]
    ).to(device)
    
    criterion = MultiClassFocalLoss(gamma=stage3_config["training"]["focal_loss_gamma"], weight=class_weights)
    
    checkpoint_dir = Path(stage3_config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    
    # ============================================================
    # RESUME TRAINING: Kiểm tra checkpoint cũ để học tiếp
    # ============================================================
    resume_checkpoint_path = checkpoint_dir / "last_checkpoint.pt"
    start_step = 1       # 1 = Step 1 (Freeze), 2 = Step 2 (Fine-tune)
    start_epoch = 0
    
    if resume_checkpoint_path.exists():
        print("🔄 Tìm thấy checkpoint cũ! Đang nạp lại kiến thức đã học...")
        resume_data = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_data['model_state_dict'])
        best_val_acc = resume_data.get('best_val_acc', 0.0)
        start_step = resume_data.get('current_step', 1)
        start_epoch = resume_data.get('current_epoch', 0) + 1  # Bắt đầu từ epoch tiếp theo
        print(f"  ✅ Đã nạp thành công! Tiếp tục từ Step {start_step}, Epoch {start_epoch + 1}")
        print(f"  ✅ Best Val Acc trước đó: {best_val_acc:.4f}")
    else:
        print("ℹ️ Không tìm thấy checkpoint cũ. Bắt đầu huấn luyện từ đầu.")
    
    # ============================================================
    # STEP 1: FREEZE VISION BACKBONE
    # ============================================================
    step1_epochs = stage3_config["training"]["step1"]["epochs"]
    
    if start_step <= 1:
        print("\n" + "="*50)
        print("📌 STEP 1: Đóng băng (Freeze) Vision Backbone")
        print(stage3_config["training"]["step1"]["description"])
        print("="*50)
        
        model.freeze_vision_branch()
        optimizer1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=stage3_config["training"]["step1"]["lr"],
            weight_decay=stage3_config["training"]["step1"]["weight_decay"]
        )
        
        # Nếu đang resume ở Step 1, nạp lại optimizer
        if resume_checkpoint_path.exists() and start_step == 1:
            if 'optimizer_state_dict' in resume_data:
                try:
                    optimizer1.load_state_dict(resume_data['optimizer_state_dict'])
                except:
                    pass
        
        epoch_start = start_epoch if start_step == 1 else 0
        for epoch in range(epoch_start, step1_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer1, device)
            val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
            print(f"Step 1 - Epoch {epoch+1}/{step1_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            # Lưu checkpoint sau MỖI epoch (chống mất dữ liệu)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            torch.save({
                'current_step': 1,
                'current_epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'best_val_acc': best_val_acc,
                'val_acc': val_acc
            }, checkpoint_dir / "last_checkpoint.pt")
        
        # Reset start_epoch cho Step 2
        start_epoch = 0
              
    # ============================================================
    # STEP 2: UNFREEZE & FINE-TUNE
    # ============================================================
    print("\n" + "="*50)
    print("📌 STEP 2: Mở khóa (Unfreeze) toàn bộ mô hình")
    print(stage3_config["training"]["step2"]["description"])
    print("="*50)
    
    model.unfreeze_vision_branch()
    optimizer2 = torch.optim.AdamW(
        model.parameters(), 
        lr=stage3_config["training"]["step2"]["lr"],
        weight_decay=stage3_config["training"]["step2"]["weight_decay"]
    )
    
    step2_epochs = stage3_config["training"]["step2"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=step2_epochs)
    
    # Nếu đang resume ở Step 2, nạp lại optimizer
    if resume_checkpoint_path.exists() and start_step == 2:
        if 'optimizer_state_dict' in resume_data:
            try:
                optimizer2.load_state_dict(resume_data['optimizer_state_dict'])
            except:
                pass
        # Tiến scheduler tới đúng epoch đã dừng
        for _ in range(start_epoch):
            scheduler.step()
    
    patience_counter = 0
    patience_limit = stage3_config["training"]["early_stopping_patience"]
    
    epoch_start = start_epoch if start_step == 2 else 0
    for epoch in range(epoch_start, step2_epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer2, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        epoch_time = time.time() - start_time
        
        print(f"Step 2 - Epoch {epoch+1}/{step2_epochs} [{epoch_time:.1f}s] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
              
        # Lưu checkpoint sau MỖI epoch (chống mất dữ liệu)
        torch.save({
            'current_step': 2,
            'current_epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer2.state_dict(),
            'best_val_acc': best_val_acc,
            'val_acc': val_acc
        }, checkpoint_dir / "last_checkpoint.pt")
        
        # Lưu best_model nếu Val Acc cải thiện
        if val_acc > best_val_acc:
            print(f"  👉 Val Acc cải thiện từ {best_val_acc:.4f} -> {val_acc:.4f}. Lưu best_model!")
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer2.state_dict(),
                'val_acc': best_val_acc
            }, checkpoint_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"⚠️ Early stopping kích hoạt tại Epoch {epoch+1}")
                break
                
    print("\n✅ Huấn luyện Stage 3 hoàn tất!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"💾 Best model lưu tại: {checkpoint_dir / 'best_model.pt'}")
    print(f"💾 Last checkpoint lưu tại: {checkpoint_dir / 'last_checkpoint.pt'}")

if __name__ == "__main__":
    # Handle multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
