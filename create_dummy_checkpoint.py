import torch
from pathlib import Path
from src.models.vision_branch import VisionBranch

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

if __name__ == "__main__":
    create_dummy_checkpoint()
