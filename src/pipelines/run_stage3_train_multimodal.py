"""
Pipeline Huấn luyện Stage 3: Mô hình Đa phương thức (Multimodal Fusion)
Áp dụng chiến lược huấn luyện 2 bước (2-Step Training)
Có hỗ trợ AMP (Mixed Precision) để tăng tốc gấp đôi trên GPU T4.
"""

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
import sys
import gc
import numpy as np
from tqdm import tqdm

from src.data.dataset import DermDataset
from src.data.augmentations import get_train_transforms, get_val_transforms
from src.data.samplers import get_weighted_sampler
from src.models.fusion_net import MultimodalFusionNet
from src.models.loss import MultiClassFocalLoss
from src.utils.seed import set_seed, get_device

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    y_a, y_b = y, y[index]
    W, H = x.size()[2], x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return x, y_a, y_b, lam

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None
    
    pbar = tqdm(dataloader, desc="Training", leave=False, file=sys.stdout)
    for images, clinicals, labels in pbar:
        images, clinicals, labels = images.to(device), clinicals.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # AMP: Chạy forward pass ở chế độ float16 để tăng tốc
        with torch.amp.autocast('cuda', enabled=use_amp):
            # CutMix augmentation (50% probability)
            if np.random.rand() < 0.5:
                images, targets_a, targets_b, lam = cutmix_data(images, labels)
                logits = model(images, clinicals)
                loss = criterion(logits, targets_a) * lam + criterion(logits, targets_b) * (1. - lam)
            else:
                logits = model(images, clinicals)
                loss = criterion(logits, labels)
        
        # AMP: Backward pass với GradScaler
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
    return running_loss / total, correct / total

def val_epoch(model, dataloader, criterion, device, use_amp=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False, file=sys.stdout)
        for images, clinicals, labels in pbar:
            images, clinicals, labels = images.to(device), clinicals.to(device), labels.to(device)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(images, clinicals)
                loss = criterion(logits, labels)
            
            running_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
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
    
    img_size = stage3_config["data"].get("img_size", 384)
    
    # Tạo 2 dataset RIÊNG BIỆT với transforms khác nhau
    # để tránh side-effect khi train/val dùng chung 1 instance
    train_dataset_full = DermDataset(
        img_dir='data/raw/images',
        is_binary=False,
        transforms=get_train_transforms(stage3_config.get("augmentations", {}), img_size=img_size),
        use_clinical=True,
        clinical_csv=stage3_config["data"]["clinical_csv_path"]
    )
    
    val_dataset_full = DermDataset(
        img_dir='data/raw/images',
        is_binary=False,
        transforms=get_val_transforms(img_size=img_size),
        use_clinical=True,
        clinical_csv=stage3_config["data"]["clinical_csv_path"]
    )
    
    # Tạo bộ chỉ số (indices) chia 80/20 một lần duy nhất
    total_len = len(train_dataset_full)
    indices = torch.randperm(total_len, generator=torch.Generator().manual_seed(base_config["seed"])).tolist()
    train_size = int(0.8 * total_len)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # 2. Tính toán class weights thực tế để chống mất cân bằng
    print("📊 Đang phân tích phân phối dữ liệu để tính Class Weights...")
    num_classes = stage3_config["model"]["num_classes"]
    
    labels = [train_dataset_full.samples[i][1] for i in train_indices]
    class_counts = torch.bincount(torch.tensor(labels), minlength=num_classes)
    total_samples = class_counts.sum().item()
    
    # Trọng số alpha = Total / (Num_Classes * Count)
    class_weights = total_samples / (num_classes * class_counts.float())
    # Giới hạn min max nếu dữ liệu quá lệch
    class_weights = torch.clamp(class_weights, min=0.1, max=10.0)
    class_weights[class_counts == 0] = 1.0  # Tránh chia cho 0
    class_weights = class_weights.to(device)
    
    print("✅ Đã tính xong Class Weights!")
    
    sampler = get_weighted_sampler(train_dataset)
    
    batch_size = stage3_config["data"]["batch_size"]
    num_workers = stage3_config["data"]["num_workers"]
    use_pin_memory = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0
    )
    
    # 3. Khởi tạo Mô hình
    print("🧠 Khởi tạo mô hình MultimodalFusionNet...")
    model = MultimodalFusionNet(
        vision_model_name=stage3_config["model"]["vision_name"],
        num_classes=stage3_config["model"]["num_classes"],
        clinical_in_features=stage3_config["model"]["clinical_in_features"],
        clinical_out_features=stage3_config["model"]["clinical_out_features"],
        pretrained=stage3_config["model"]["pretrained"]
    )
    
    if torch.cuda.device_count() > 1:
        print(f"🚀 Tìm thấy {torch.cuda.device_count()} GPUs! Kích hoạt chế độ huấn luyện Multi-GPU (DataParallel)...")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
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
    # KÍCH HOẠT AMP (Mixed Precision) - Tăng tốc gấp đôi trên GPU T4
    # ============================================================
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    if use_amp:
        print("⚡ AMP (Mixed Precision) đã được kích hoạt! Tốc độ tăng gấp 1.5-2x.")
    sys.stdout.flush()
    
    # ============================================================
    # STEP 1: FREEZE VISION BACKBONE
    # ============================================================
    step1_epochs = stage3_config["training"]["step1"]["epochs"]
    
    if start_step <= 1:
        print("\n" + "="*50)
        print("📌 STEP 1: Đóng băng (Freeze) Vision Backbone")
        print(stage3_config["training"]["step1"]["description"])
        print("="*50)
        
        if isinstance(model, nn.DataParallel):
            model.module.freeze_vision_branch()
        else:
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
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer1, device, scaler)
            val_loss, val_acc = val_epoch(model, val_loader, criterion, device, use_amp)
            print(f"Step 1 - Epoch {epoch+1}/{step1_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            sys.stdout.flush()
            
            # Lưu checkpoint sau MỖI epoch (chống mất dữ liệu)
            if val_acc > best_val_acc:
                print(f"  👉 Val Acc cải thiện lên {val_acc:.4f}. Lưu best_model!")
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer1.state_dict(),
                    'val_acc': best_val_acc
                }, checkpoint_dir / "best_model.pt")
                
            torch.save({
                'current_step': 1,
                'current_epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'best_val_acc': best_val_acc,
                'val_acc': val_acc
            }, checkpoint_dir / "last_checkpoint.pt")
            
            # Dọn rác RAM sau mỗi epoch
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Reset start_epoch cho Step 2
        start_epoch = 0
              
    # ============================================================
    # STEP 2: UNFREEZE & FINE-TUNE
    # ============================================================
    print("\n" + "="*50)
    print("📌 STEP 2: Mở khóa (Unfreeze) toàn bộ mô hình")
    print(stage3_config["training"]["step2"]["description"])
    print("="*50)
    
    if isinstance(model, nn.DataParallel):
        model.module.unfreeze_vision_branch()
    else:
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
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer2, device, scaler)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device, use_amp)
        
        scheduler.step()
        epoch_time = time.time() - start_time
        
        print(f"Step 2 - Epoch {epoch+1}/{step2_epochs} [{epoch_time:.1f}s] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        sys.stdout.flush()
              
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
            sys.stdout.flush()
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"⚠️ Early stopping kích hoạt tại Epoch {epoch+1}")
                break
        
        # Dọn rác RAM sau mỗi epoch
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
                
    print("\n✅ Huấn luyện Stage 3 hoàn tất!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"💾 Best model lưu tại: {checkpoint_dir / 'best_model.pt'}")
    print(f"💾 Last checkpoint lưu tại: {checkpoint_dir / 'last_checkpoint.pt'}")
    sys.stdout.flush()

if __name__ == "__main__":
    # Handle multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
