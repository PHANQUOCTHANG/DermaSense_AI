"""
DermaSense AI — Tests for Stage 2
"""
import torch
import numpy as np
from src.models.loss import BinaryFocalLoss
from src.models.vision_branch import VisionBranch

def test_binary_focal_loss():
    loss_fn = BinaryFocalLoss(alpha=0.25, gamma=2.0)
    
    # 2 mẫu, 2 lớp
    logits = torch.tensor([[1.5, -0.5], [-1.0, 2.0]])
    targets = torch.tensor([0, 1])
    
    loss = loss_fn(logits, targets)
    
    assert loss is not None
    assert not torch.isnan(loss)
    assert loss.item() > 0

def test_vision_branch_output_shape():
    model = VisionBranch(num_classes=2, pretrained=False)
    
    # Batch = 2, Channels = 3, Size = 384x384
    dummy_input = torch.randn(2, 3, 384, 384)
    
    output = model(dummy_input)
    
    assert output.shape == (2, 2)
    assert not torch.isnan(output).any()
