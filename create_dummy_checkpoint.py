import torch
from pathlib import Path
from src.models.vision_branch import VisionBranch
from src.models.fusion_net import MultimodalFusionNet

def create_dummy_checkpoint():
    print("Creating dummy checkpoint for demo purposes...")
    
    # Initialize the model with random weights
    model = VisionBranch(
        model_name="tf_efficientnetv2_s.in21k_ft_in1k", 
        num_classes=2, 
        pretrained=False
    )
    
    # Dummy metrics
    metrics = {
        'sensitivity': 0.96,
        'npv': 0.98,
        'auc': 0.95
    }
    
    checkpoint_dir = Path("models/checkpoints/stage_a")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"
    
    torch.save({
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'metrics': metrics
    }, checkpoint_path)
    print(f"Dummy checkpoint saved at: {checkpoint_path}")
    
    # Dummy Stage 3
    model3 = MultimodalFusionNet(
        vision_model_name="tf_efficientnetv2_m.in21k_ft_in1k",
        num_classes=23,
        clinical_in_features=7,
        pretrained=False
    )
    
    checkpoint_dir3 = Path("models/checkpoints/stage_b")
    checkpoint_dir3.mkdir(parents=True, exist_ok=True)
    checkpoint_path3 = checkpoint_dir3 / "best_model.pt"
    
    torch.save({
        'epoch': 1,
        'model_state_dict': model3.state_dict(),
        'metrics': {'acc': 0.85}
    }, checkpoint_path3)
    
    print(f"Dummy Stage 3 checkpoint saved at: {checkpoint_path3}")

if __name__ == "__main__":
    create_dummy_checkpoint()
