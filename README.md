# 🩺 DermaSense AI

**Hệ thống Sàng lọc Da liễu Đa tầng, Đa phương thức, Khả giải**

> Multi-stage, Multi-modal, Explainable Dermatology Screening System

---

## 📋 Tổng quan

DermaSense AI là hệ thống AI hỗ trợ sàng lọc bệnh da liễu theo kiến trúc đa tầng:

- **Stage A — Safety Screener:** Phân loại nhị phân "cần chú ý / không đáng lo" với sensitivity cực cao (>0.95)
- **Stage B — Multimodal Classifier:** Phân loại chi tiết 7 lớp bệnh học, kết hợp ảnh da liễu + dữ liệu lâm sàng
- **XAI Layer:** Giải thích dự đoán bằng Grad-CAM++, hiệu chuẩn xác suất (Temperature Scaling), phát hiện ảnh ngoài phân phối (OOD)
- **Stage C — Human-in-the-loop:** Vòng lặp xác nhận bởi bác sĩ cho các case nguy cơ cao

## 🏗️ Kiến trúc hệ thống

```
Input Image → Preprocessing Pipeline → Stage A (Safety Screener)
                                            │
                                    ┌───────┴───────┐
                                    │               │
                              "Không đáng lo"   "Cần chú ý"
                              → Kết thúc         → Stage B
                                                     │
                                              ┌──────┴──────┐
                                              │             │
                                         Vision Branch  Clinical Branch
                                         (EfficientNetV2)  (MLP)
                                              │             │
                                              └──────┬──────┘
                                                     │
                                                Fusion Layer
                                                     │
                                              7-class prediction
                                                     │
                                              Risk Mapping (3 levels)
                                                     │
                                              XAI (Grad-CAM++ + Calibration)
                                                     │
                                        ┌────────────┴────────────┐
                                        │                        │
                                  Nguy cơ thấp/TB          Nguy cơ cao
                                  → Trả kết quả        → Stage C (Bác sĩ review)
```

## 📁 Cấu trúc thư mục

```
DermaSense_AI/
├── configs/                    # Cấu hình YAML cho từng giai đoạn
│   ├── base_config.yaml        # Config chung (paths, seed, device)
│   ├── stage1_preprocessing.yaml
│   ├── stage2_balancing.yaml
│   ├── stage3_train_multimodal.yaml
│   └── stage4_xai_eval.yaml
├── data/
│   ├── raw/                    # Dữ liệu gốc (ISIC/HAM10000)
│   │   ├── images/
│   │   └── metadata.csv
│   ├── processed/              # Dữ liệu đã tiền xử lý
│   └── clinical/               # Dữ liệu lâm sàng (tuổi, giới, vị trí...)
├── models/
│   ├── checkpoints/            # Model weights (.pt)
│   ├── baselines/              # Baseline models để so sánh
│   └── exported/               # ONNX exported models
├── notebooks/                  # Jupyter notebooks (chạy trên Kaggle)
│   ├── 00_eda_dataset_distribution.ipynb
│   ├── 01_test_preprocessing_pipeline.ipynb
│   ├── 02_check_class_imbalance.ipynb
│   └── 03_visualize_xai_gradcam.ipynb
├── outputs/
│   ├── figures/                # Biểu đồ, confusion matrix, ROC
│   ├── logs/                   # Training logs
│   └── xai_heatmaps/          # Grad-CAM++ heatmaps
├── src/
│   ├── data/                   # Data loading, preprocessing, augmentation
│   ├── models/                 # Model architectures
│   ├── pipelines/              # Training & evaluation scripts
│   ├── utils/                  # Utilities (logger, metrics, seed)
│   └── xai/                    # Explainability modules
├── tests/                      # Unit tests
├── .env                        # API keys (KHÔNG commit)
├── .gitignore
├── check_gpu.py                # Kiểm tra môi trường GPU
├── dvc.yaml                    # DVC pipeline definition
└── requirements.txt            # Python dependencies
```

## 🔧 Cài đặt

### Môi trường local (code only)
```bash
git clone <repo-url>
cd DermaSense_AI
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Huấn luyện trên Kaggle Notebooks (miễn phí GPU)
1. Upload repo lên Kaggle Dataset hoặc kết nối GitHub
2. Tạo Notebook mới, chọn **GPU T4 x2** accelerator
3. Mount dataset ISIC 2019 từ Kaggle: `kaggle datasets download -d andrewmvd/isic-2019`
4. Cài thêm dependencies: `!pip install -r requirements.txt`

## 🗃️ Dữ liệu

- **ISIC 2019/2020** — ~25,000 ảnh da liễu, 8 lớp bệnh học
- **HAM10000** — 10,015 ảnh, 7 lớp
- Nguồn: [ISIC Archive](https://www.isic-archive.com/) | [Kaggle ISIC 2019](https://www.kaggle.com/datasets/andrewmvd/isic-2019)

## 📊 Chỉ tiêu hiệu năng

| Metric | Mục tiêu |
|---|---|
| Stage A — Sensitivity | > 0.95 |
| Stage A — NPV | > 0.97 |
| Stage B — AUC-ROC (macro) | > 0.90 |
| Stage B — Melanoma Sensitivity | > 0.90 |
| Stage B — F1-score (weighted) | > 0.82 |
| XAI — ECE sau hiệu chuẩn | < 0.05 |

## 🛠️ Công nghệ chính

| Nhóm | Công nghệ |
|---|---|
| Deep Learning | PyTorch 2.x, PyTorch Lightning, timm |
| Vision Backbone | EfficientNetV2-S/M |
| Xử lý ảnh | OpenCV, scikit-image, Albumentations |
| XAI | pytorch-grad-cam, Temperature Scaling |
| Experiment Tracking | Weights & Biases |
| Data Versioning | DVC |
| Cloud Training | **Kaggle Notebooks** (GPU T4 miễn phí) |

## 📜 License

Dự án phục vụ mục đích nghiên cứu khoa học. Mọi dữ liệu bệnh nhân đã được ẩn danh hoá theo quy định đạo đức nghiên cứu y sinh.
