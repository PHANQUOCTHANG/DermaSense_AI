"""
DermaSense AI — Seed Utility
Đảm bảo reproducibility cho toàn bộ pipeline.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Thiết lập seed cho tất cả thư viện để đảm bảo kết quả reproducible.

    Args:
        seed: Giá trị seed (mặc định 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Nếu dùng multi-GPU

    # Đảm bảo deterministic trên CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_device(preference: str = "auto") -> torch.device:
    """Tự động detect device tốt nhất có sẵn.

    Args:
        preference: "auto" | "cuda" | "cpu"
                    - auto: ưu tiên CUDA nếu có, fallback CPU
                    - cuda: ép dùng CUDA (lỗi nếu không có)
                    - cpu: ép dùng CPU

    Returns:
        torch.device phù hợp.
    """
    if preference == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available!")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device
