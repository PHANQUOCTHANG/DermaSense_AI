"""
DermaSense AI — Kiểm tra Môi trường
Chạy script này để xác nhận setup đã hoàn tất.
Hoạt động trên cả local (CPU) lẫn Kaggle Notebooks (GPU).

Usage:
    python check_gpu.py
    # Hoặc trên Kaggle: !python check_gpu.py
"""
import importlib
import os
import sys
from pathlib import Path


def check_section(title: str) -> None:
    """In header cho mỗi section kiểm tra."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


def check_python() -> bool:
    """Kiểm tra phiên bản Python."""
    check_section("🐍 PYTHON")
    v = sys.version_info
    print(f"  Phiên bản: {sys.version}")
    ok = v.major == 3 and v.minor >= 10
    status = "✅ OK" if ok else "⚠️ Khuyến nghị Python 3.10+"
    print(f"  {status}")
    return ok


def check_pytorch() -> bool:
    """Kiểm tra PyTorch và CUDA."""
    check_section("⚡ PYTORCH & CUDA")
    try:
        import torch

        print(f"  PyTorch version: {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_ok}")

        if cuda_ok:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Số GPU: {torch.cuda.device_count()}")
            mem = torch.cuda.get_device_properties(0).total_mem
            print(f"  VRAM: {mem / 1e9:.1f} GB")
            # Quick test
            x = torch.rand(3, 3, device="cuda")
            _ = x @ x.T
            print("  ✅ GPU tensor test: PASSED")
        else:
            print("  ℹ️ Không có GPU — sẽ dùng CPU (đủ để code & test)")
            print("  ℹ️ Huấn luyện nên chạy trên Kaggle Notebooks (GPU T4 miễn phí)")

        return True
    except ImportError:
        print("  ❌ PyTorch chưa cài! Chạy: pip install torch torchvision")
        return False


def check_dependencies() -> dict:
    """Kiểm tra các thư viện quan trọng."""
    check_section("📦 DEPENDENCIES")

    packages = {
        "torchvision": "torchvision",
        "timm": "timm",
        "pytorch_lightning": "pytorch_lightning",
        "cv2": "opencv-python",
        "albumentations": "albumentations",
        "sklearn": "scikit-learn",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "yaml": "pyyaml",
        "wandb": "wandb",
        "PIL": "Pillow",
        "tqdm": "tqdm",
        "skimage": "scikit-image",
    }

    results = {}
    for import_name, pip_name in packages.items():
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            print(f"  ✅ {pip_name:<22s} v{version}")
            results[pip_name] = True
        except ImportError:
            print(f"  ❌ {pip_name:<22s} — pip install {pip_name}")
            results[pip_name] = False

    return results


def check_project_structure() -> bool:
    """Kiểm tra cấu trúc thư mục dự án."""
    check_section("📁 CẤU TRÚC DỰ ÁN")

    # Detect project root
    # Trên Kaggle: /kaggle/working/DermaSense_AI hoặc mount từ dataset
    # Trên local: thư mục chứa script này
    project_root = Path(__file__).parent

    required_dirs = [
        "configs",
        "data/raw",
        "data/processed",
        "data/clinical",
        "models/checkpoints",
        "models/baselines",
        "models/exported",
        "notebooks",
        "outputs/figures",
        "outputs/logs",
        "outputs/xai_heatmaps",
        "src/data",
        "src/models",
        "src/pipelines",
        "src/utils",
        "src/xai",
        "tests",
    ]

    required_files = [
        "configs/base_config.yaml",
        "configs/stage1_preprocessing.yaml",
        "configs/stage2_balancing.yaml",
        "configs/stage3_train_multimodal.yaml",
        "configs/stage4_xai_eval.yaml",
        "requirements.txt",
        "README.md",
        ".gitignore",
    ]

    all_ok = True

    print("  --- Thư mục ---")
    for d in required_dirs:
        path = project_root / d
        exists = path.is_dir()
        status = "✅" if exists else "❌"
        print(f"  {status} {d}/")
        if not exists:
            all_ok = False

    print("\n  --- File cấu hình ---")
    for f in required_files:
        path = project_root / f
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        empty = " (⚠️ rỗng)" if exists and size == 0 else ""
        status = "✅" if exists and size > 0 else ("⚠️" if exists else "❌")
        print(f"  {status} {f}{empty}")
        if not exists:
            all_ok = False

    return all_ok


def check_kaggle_env() -> bool:
    """Kiểm tra xem đang chạy trên Kaggle hay không."""
    check_section("☁️ MÔI TRƯỜNG KAGGLE")

    is_kaggle = os.path.exists("/kaggle")
    if is_kaggle:
        print("  ✅ Đang chạy trên Kaggle Notebooks")
        print(f"  📂 Input: /kaggle/input/")
        print(f"  📂 Working: /kaggle/working/")

        # Kiểm tra dataset ISIC
        isic_path = Path("/kaggle/input/isic-2019")
        if isic_path.exists():
            num_files = sum(1 for _ in isic_path.rglob("*") if _.is_file())
            print(f"  ✅ ISIC 2019 dataset found ({num_files} files)")
        else:
            print("  ℹ️ ISIC 2019 chưa mount — Add Dataset: andrewmvd/isic-2019")
    else:
        print("  ℹ️ Đang chạy local (không phải Kaggle)")
        print("  ℹ️ Để huấn luyện: upload code lên Kaggle, chọn GPU T4")

    return is_kaggle


def main() -> None:
    """Chạy toàn bộ kiểm tra."""
    print("🩺 DermaSense AI — Environment Check")
    print("=" * 50)

    results = {}
    results["python"] = check_python()
    results["pytorch"] = check_pytorch()
    results["deps"] = check_dependencies()
    results["structure"] = check_project_structure()
    results["kaggle"] = check_kaggle_env()

    # Summary
    check_section("📊 TỔNG KẾT")

    all_critical_ok = results["python"] and results["pytorch"] and results["structure"]

    if all_critical_ok:
        print("  ✅ Giai đoạn 0 — Setup: HOÀN TẤT")
        print("  ℹ️ Sẵn sàng chuyển sang Giai đoạn 1 (Tiền xử lý)")
    else:
        print("  ⚠️ Còn một số vấn đề cần khắc phục (xem chi tiết ở trên)")

    missing_deps = [k for k, v in results.get("deps", {}).items() if not v]
    if missing_deps:
        print(f"\n  Cài thêm: pip install {' '.join(missing_deps)}")


if __name__ == "__main__":
    main()