"""
DermaSense AI — Pipeline Stage 2: Huấn luyện Safety Screener (Binary Classification)
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.data.augmentations import get_train_transforms, get_val_transforms
from src.data.dataset import DermDataset
from src.data.samplers import get_weighted_sampler
from src.models.loss import BinaryFocalLoss
from src.models.vision_branch import VisionBranch
from src.utils.logger import get_logger
from src.utils.metrics import compute_classification_metrics, compute_sensitivity_specificity
from src.utils.seed import get_device, set_seed


def load_configs():
    with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    with open("configs/stage2_balancing.yaml", "r", encoding="utf-8") as f:
        stage2_balancing = yaml.safe_load(f)
    with open("configs/stage2_train_screener.yaml", "r", encoding="utf-8") as f:
        stage2_train = yaml.safe_load(f)
    return base_config, stage2_balancing, stage2_train


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Validating", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return epoch_loss, all_labels, all_preds, all_probs


def main():
    base_cfg, bal_cfg, train_cfg = load_configs()
    paths = base_cfg["paths"]
    
    logger = get_logger("stage2_train", log_dir=paths["logs"])
    logger.info("="*50)
    logger.info("Start Stage 2: Training Safety Screener (Binary)")
    logger.info("="*50)
    
    set_seed(base_cfg["seed"])
    device = get_device(base_cfg["device"])
    logger.info(f"Using device: {device}")
    
    # 1. Chuẩn bị Data
    processed_dir = Path(paths["data_processed"])
    
    train_transforms = get_train_transforms(bal_cfg["augmentation"])
    val_transforms = get_val_transforms()
    
    train_dataset = DermDataset(img_dir=str(processed_dir / "train"), is_binary=True, transforms=train_transforms)
    val_dataset = DermDataset(img_dir=str(processed_dir / "val"), is_binary=True, transforms=val_transforms)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    if bal_cfg["sampler"]["enabled"]:
        train_sampler = get_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=train_cfg["training"]["batch_size"], 
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=train_cfg["training"]["batch_size"], 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
    val_loader = DataLoader(
        val_dataset, 
        batch_size=train_cfg["training"]["batch_size"], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 2. Chuẩn bị Model & Loss
    model = VisionBranch(
        model_name=train_cfg["model"]["name"],
        num_classes=2,
        pretrained=train_cfg["model"]["pretrained"],
        drop_rate=train_cfg["model"]["drop_rate"]
    ).to(device)
    
    criterion = BinaryFocalLoss(
        gamma=bal_cfg["focal_loss"]["gamma"],
        alpha=bal_cfg["focal_loss"]["alpha"]
    )
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=train_cfg["training"]["learning_rate"], 
        weight_decay=train_cfg["training"]["weight_decay"]
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=train_cfg["training"]["epochs"]
    )
    
    # 3. Vòng lặp Huấn luyện
    epochs = train_cfg["training"]["epochs"]
    best_sensitivity = 0.0
    patience_counter = 0
    patience = train_cfg["training"]["early_stopping_patience"]
    
    checkpoint_dir = Path(paths["checkpoints"]) / "stage_a"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.pt"
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_labels, val_preds, val_probs = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Đánh giá Metrics
        metrics = compute_classification_metrics(val_labels, val_preds, val_probs)
        sens_spec = compute_sensitivity_specificity(val_labels, val_preds)
        
        val_sensitivity = sens_spec["sensitivity"]
        val_npv = sens_spec["npv"]
        val_auc = metrics.get("auc_roc", 0)
        
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.info(f"Val Sens: {val_sensitivity:.4f} | Val NPV: {val_npv:.4f} | Val AUC: {val_auc:.4f}")
        
        # Lưu best model dựa trên Sensitivity (Ưu tiên số 1 của Safety Screener)
        if val_sensitivity > best_sensitivity:
            best_sensitivity = val_sensitivity
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {
                    'sensitivity': val_sensitivity,
                    'npv': val_npv,
                    'auc': val_auc
                }
            }, best_model_path)
            logger.info(f"-> Saved new best model with Sensitivity: {best_sensitivity:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    logger.info("="*50)
    logger.info("Stage 2 Training Completed!")
    logger.info(f"Best Val Sensitivity: {best_sensitivity:.4f}")

if __name__ == "__main__":
    main()
