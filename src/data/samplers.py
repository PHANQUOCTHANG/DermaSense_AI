"""
DermaSense AI — Samplers Utility
Giải quyết bài toán mất cân bằng dữ liệu bằng WeightedRandomSampler.
"""
from typing import List
import torch
from torch.utils.data import WeightedRandomSampler

def get_weighted_sampler(dataset) -> WeightedRandomSampler:
    """Tạo WeightedRandomSampler dựa trên phân phối của dataset.
    
    Args:
        dataset: Đối tượng DermDataset đã được khởi tạo.
        
    Returns:
        WeightedRandomSampler để nạp vào DataLoader.
    """
    # Lấy toàn bộ nhãn từ tập dữ liệu
    labels = [sample[1] for sample in dataset.samples]
    
    # Đếm số lượng mẫu của mỗi lớp
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
        
    # Tính trọng số cho từng lớp (nghịch đảo tần suất)
    num_samples = len(labels)
    class_weights = {
        cls: num_samples / count 
        for cls, count in class_counts.items()
    }
    
    # Gán trọng số cho từng mẫu cụ thể
    sample_weights = [class_weights[label] for label in labels]
    
    # Tạo sampler (lấy mẫu có hoàn lại - replacement=True)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )
    
    return sampler
