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
from src.data.samplers import get_balanced_sampler
from src.models.fusion_net import MultimodalFusionNet
from src.models.loss import BinaryFocalLoss # Sẽ cần điều chỉnh hoặc dùng CrossEntropy, nhưng vì 7 classes nên sẽ dùng nn.CrossEntropyLoss hoặc Focal Loss đa lớp
from src.utils.seed import set_seed, get_device

# Tạo hàm Focal Loss cho Đa Lớp
class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

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
    train_dataset = DermDataset(
        img_dir=f'{base_config["paths"]["data_processed"]}/train',
        is_binary=False,
        transforms=get_train_transforms(),
        use_clinical=True,
        clinical_csv=stage3_config["data"]["clinical_csv_path"]
    )
    
    val_dataset = DermDataset(
        img_dir=f'{base_config["paths"]["data_processed"]}/val',
        is_binary=False,
        transforms=get_val_transforms(),
        use_clinical=True,
        clinical_csv=stage3_config["data"]["clinical_csv_path"]
    )
    
    # Dummy weight cho 7 classes để tránh lỗi nếu thư mục không đủ ảnh
    class_weights = torch.ones(7).to(device)
    sampler, _ = get_balanced_sampler(train_dataset, num_classes=7)
    
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
    # STEP 1: FREEZE VISION BACKBONE
    # ============================================================
    print("\n" + "="*50)
    print("📌 STEP 1: Đóng băng (Freeze) Vision Backbone")
    print(stage3_config["training"]["step1"]["description"])
    print("="*50)
    
    model.freeze_vision_branch()
    # Chỉ tối ưu Clinical Branch và Fusion Head
    optimizer1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=stage3_config["training"]["step1"]["lr"],
        weight_decay=stage3_config["training"]["step1"]["weight_decay"]
    )
    
    step1_epochs = stage3_config["training"]["step1"]["epochs"]
    for epoch in range(step1_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer1, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        print(f"Step 1 - Epoch {epoch+1}/{step1_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
              
    # ============================================================
    # STEP 2: UNFREEZE & FINE-TUNE
    # ============================================================
    print("\n" + "="*50)
    print("📌 STEP 2: Mở khóa (Unfreeze) toàn bộ mô hình")
    print(stage3_config["training"]["step2"]["description"])
    print("="*50)
    
    model.unfreeze_vision_branch()
    # Tối ưu toàn bộ mô hình
    optimizer2 = torch.optim.AdamW(
        model.parameters(), 
        lr=stage3_config["training"]["step2"]["lr"],
        weight_decay=stage3_config["training"]["step2"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=stage3_config["training"]["step2"]["epochs"])
    
    step2_epochs = stage3_config["training"]["step2"]["epochs"]
    patience_counter = 0
    patience_limit = stage3_config["training"]["early_stopping_patience"]
    
    for epoch in range(step2_epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer2, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        epoch_time = time.time() - start_time
        
        print(f"Step 2 - Epoch {epoch+1}/{step2_epochs} [{epoch_time:.1f}s] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
              
        # Lưu checkpoint nếu Val Acc tăng
        if val_acc > best_val_acc:
            print(f"  👉 Val Acc cải thiện từ {best_val_acc:.4f} -> {val_acc:.4f}. Lưu mô hình!")
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
    print(f"💾 Checkpoint lưu tại: {checkpoint_dir / 'best_model.pt'}")

if __name__ == "__main__":
    # Handle multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
