# 🩺 DermaSense AI — Sơ đồ Tổng quan Hệ thống

---

## 1. Tổng quan Pipeline — 4 Giai đoạn chính

Hệ thống DermaSense AI hoạt động theo **4 giai đoạn tuần tự**, từ dữ liệu thô đến kết quả chẩn đoán có giải thích.

```mermaid
flowchart TB
    subgraph STAGE0["⚙️ STAGE 0 — Setup & Kiểm tra Môi trường"]
        S0A["check_gpu.py"] --> S0B["Cài đặt dependencies<br/>(requirements.txt)"]
        S0B --> S0C["Xác nhận GPU/CPU<br/>& cấu trúc dự án"]
    end

    subgraph STAGE1["🧹 STAGE 1 — Tiền xử lý Dữ liệu"]
        S1A["Dữ liệu thô<br/>(ISIC 2019, HAM10000)"] --> S1B["Phát hiện ảnh mờ<br/>(Laplacian Variance)"]
        S1B --> S1C["Xóa lông<br/>(DullRazor)"]
        S1C --> S1D["Cân bằng màu<br/>(Gray World)"]
        S1D --> S1E["Resize 384×384<br/>& Normalize"]
        S1E --> S1F["Chia train/val/test<br/>(70/15/15 Stratified)"]
    end

    subgraph STAGE2["⚖️ STAGE 2 — Cân bằng Dữ liệu"]
        S2A["Phân tích phân bố<br/>7 lớp bệnh"] --> S2B["WeightedRandomSampler<br/>(Inverse Frequency)"]
        S2A --> S2C["Focal Loss<br/>(gamma=2)"]
        S2A --> S2D["Augmentation nâng cao<br/>(CutMix, Elastic Transform)"]
    end

    subgraph STAGE3["🧠 STAGE 3 — Huấn luyện Mô hình"]
        S3A["Stage A: Binary Screener<br/>(High Risk vs Low Risk)"]
        S3B["Stage B: Multimodal 7-class<br/>(EfficientNetV2 + Clinical MLP)"]
        S3A --> S3B
    end

    subgraph STAGE4["📊 STAGE 4 — Đánh giá & Giải thích"]
        S4A["Tính Metrics<br/>(AUC-ROC, F1, Sensitivity)"]
        S4B["Temperature Scaling<br/>(Hiệu chuẩn xác suất)"]
        S4C["Grad-CAM++ Heatmap<br/>(Giải thích quyết định)"]
        S4D["Export ONNX<br/>(Deploy)"]
    end

    STAGE0 ==> STAGE1
    STAGE1 ==> STAGE2
    STAGE2 ==> STAGE3
    STAGE3 ==> STAGE4

    style STAGE0 fill:#1a1a2e,stroke:#e94560,color:#fff
    style STAGE1 fill:#16213e,stroke:#0f3460,color:#fff
    style STAGE2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style STAGE3 fill:#0f3460,stroke:#533483,color:#fff
    style STAGE4 fill:#16213e,stroke:#e94560,color:#fff
```

> **Giải thích:** Hệ thống được chia thành 4 giai đoạn rõ ràng. Mỗi giai đoạn có file cấu hình YAML riêng (`configs/stageX_*.yaml`), script chạy riêng (`src/pipelines/run_stageX_*.py`), và output riêng. Điều này đảm bảo tính **reproducibility** (lặp lại được) và dễ dàng debug từng bước.

---

## 2. Luồng Tiền xử lý Dữ liệu (Stage 1) — Chi tiết

```mermaid
flowchart LR
    subgraph INPUT["📥 Input"]
        RAW["data/raw/images/<br/>Ảnh da liễu gốc<br/>(ISIC 2019/2020)"]
        META["data/raw/metadata.csv<br/>(age, sex, site, diagnosis)"]
    end

    subgraph PREPROCESS["🧹 Pipeline Tiền xử lý"]
        direction TB
        P1["1️⃣ Laplacian Variance<br/>────────────────<br/>Tính phương sai gradient<br/>var < 100 → ảnh mờ → gắn cờ"]
        P2["2️⃣ DullRazor<br/>────────────────<br/>Blackhat morphology → detect lông<br/>Binary threshold → Inpainting"]
        P3["3️⃣ Gray World<br/>────────────────<br/>Cân bằng kênh RGB<br/>Loại bỏ color cast"]
        P4["4️⃣ Resize & Normalize<br/>────────────────<br/>384×384 bilinear<br/>ImageNet mean/std"]
        P1 --> P2 --> P3 --> P4
    end

    subgraph SPLIT["✂️ Chia dữ liệu"]
        SP["Stratified Split<br/>70% Train / 15% Val / 15% Test"]
    end

    subgraph OUTPUT["📤 Output"]
        OUT1["data/processed/train/"]
        OUT2["data/processed/val/"]
        OUT3["data/processed/test/"]
        OUT4["data/clinical/<br/>metadata_encoded.csv"]
        OUT5["data/processed/rejected/<br/>Ảnh bị loại (review)"]
    end

    RAW --> P1
    META --> SP
    P4 --> SP
    SP --> OUT1
    SP --> OUT2
    SP --> OUT3
    SP --> OUT4
    P1 -.->|"ảnh mờ"| OUT5
```

> **Giải thích từng bước:**
> - **Laplacian Variance:** Phát hiện ảnh bị mờ (chất lượng kém) bằng cách tính phương sai của đạo hàm bậc 2 Laplacian. Ảnh mờ sẽ bị gắn cờ hoặc loại ra.
> - **DullRazor:** Thuật toán chuyên dụng cho ảnh da liễu — phát hiện và xóa lông trên da bằng phép toán hình thái học (morphological blackhat) kết hợp inpainting.
> - **Gray World:** Cân bằng trắng để các ảnh chụp từ các thiết bị/ánh sáng khác nhau có tông màu nhất quán.
> - **Resize + Normalize:** Đưa tất cả ảnh về kích thước 384×384 (input chuẩn của EfficientNetV2) và chuẩn hóa theo thống kê ImageNet.

---

## 3. Kiến trúc Mô hình — 2 Stage Multimodal

```mermaid
flowchart TB
    subgraph STAGE_A["🛡️ STAGE A — Safety Screener (Binary)"]
        direction LR
        IA["Ảnh da liễu<br/>384×384"] --> VA["EfficientNetV2-S<br/>(Pretrained ImageNet)"]
        VA --> FA["Feature 1280-d"]
        ABCDE["ABCDE Features<br/>(Asymmetry, Border,<br/>Color, Diameter,<br/>Evolution)"] --> MLPA["MLP 128-d"]
        FA --> CONCAT_A["Concat"]
        MLPA --> CONCAT_A
        CONCAT_A --> CLA["Classifier<br/>FC → 2 classes"]
        CLA --> OUTA{{"High Risk ⚠️<br/>vs Low Risk ✅"}}
    end

    subgraph STAGE_B["🔬 STAGE B — Multimodal 7-class Classifier"]
        direction LR
        IB["Ảnh da liễu<br/>384×384"] --> VB["EfficientNetV2-M<br/>(Vision Branch)"]
        VB --> FB["Feature 1280-d"]
        
        CB["Clinical Metadata<br/>(age, sex, site,<br/>duration, symptoms)"] --> EMB["Embedding +<br/>StandardScaler"]
        EMB --> MLPB["Clinical MLP<br/>128 → 64"]
        
        FB --> FUSION["Fusion Layer<br/>(Concat / Gating)"]
        MLPB --> FUSION
        FUSION --> FC["FC 256 → 7"]
        FC --> OUTB{{"MEL | NV | BCC<br/>AK | BKL | DF | VASC"}}
    end

    OUTA -->|"High Risk"| STAGE_B
    OUTA -->|"Low Risk"| SAFE["✅ Lành tính<br/>Không cần khám thêm"]

    style STAGE_A fill:#2d1b69,stroke:#e94560,color:#fff
    style STAGE_B fill:#1b3a4b,stroke:#0f3460,color:#fff
    style SAFE fill:#1b4332,stroke:#2d6a4f,color:#fff
```

> **Giải thích kiến trúc 2 tầng:**
> - **Stage A (Safety Screener):** Bộ sàng lọc nhị phân — nhanh chóng phân loại ảnh thành "Cần chú ý" (melanoma, BCC, AK) vs "Lành tính". Yêu cầu **Sensitivity ≥ 95%** để không bỏ sót ca nguy hiểm. Kết hợp đặc trưng ABCDE (Bất đối xứng, Đường biên, Màu sắc, Đường kính, Tiến triển) với ảnh.
> - **Stage B (Multimodal Classifier):** Chỉ chạy với các ca "Cần chú ý" → phân loại chi tiết thành 7 loại bệnh. Kết hợp **Vision Branch** (EfficientNetV2-M trích xuất đặc trưng ảnh) với **Clinical Branch** (MLP xử lý metadata lâm sàng như tuổi, giới tính, vị trí tổn thương) thông qua **Fusion Layer**.

---

## 4. Chiến lược Huấn luyện (Stage 3) — Chi tiết

```mermaid
flowchart TB
    subgraph TRAIN["🧠 Training Pipeline"]
        direction TB
        
        subgraph DATA_LOAD["📦 Data Loading"]
            DL1["DermDataset<br/>(Ảnh + Clinical CSV)"]
            DL2["WeightedRandomSampler<br/>(Cân bằng lớp thiểu số)"]
            DL3["Augmentations<br/>(Flip, Rotate, CutMix,<br/>Elastic Transform)"]
            DL1 --> DL2 --> DL3
        end

        subgraph STEP1["Step 1: Freeze Backbone (5 epochs)"]
            F1["🔒 Freeze EfficientNetV2<br/>(Không cập nhật weights)"]
            F2["🔓 Chỉ train Fusion + Classifier<br/>(LR = 1e-3)"]
            F1 --> F2
        end

        subgraph STEP2["Step 2: Unfreeze & Fine-tune (45 epochs)"]
            U1["🔓 Unfreeze toàn bộ<br/>(Fine-tune backbone)"]
            U2["Cosine Annealing LR<br/>(5e-5 → 0)"]
            U3["Early Stopping<br/>(patience = 10)"]
            U1 --> U2 --> U3
        end

        subgraph LOSS_FN["⚖️ Loss Function"]
            LF["Focal Loss (γ=2)<br/>+ Class Weights α"]
        end

        DATA_LOAD --> STEP1
        STEP1 --> STEP2
        LOSS_FN --> STEP1
        LOSS_FN --> STEP2
    end

    subgraph MONITOR["📈 Monitoring"]
        M1["W&B / TensorBoard<br/>Loss, Accuracy, AUC per epoch"]
        M2["Best Checkpoint Saving<br/>(val_auc_roc tốt nhất)"]
    end

    STEP2 --> M1
    STEP2 --> M2
    M2 --> CKPT["models/checkpoints/<br/>best_model.pt"]

    style TRAIN fill:#1a1a2e,stroke:#e94560,color:#fff
    style MONITOR fill:#16213e,stroke:#0f3460,color:#fff
```

> **Giải thích chiến lược 2-step training:**
> - **Step 1 (Freeze):** Đóng băng backbone EfficientNetV2 (đã pretrained trên ImageNet), chỉ huấn luyện lớp Fusion và Classifier. Mục đích: để các lớp mới học được cách "đọc" feature từ backbone mà không làm hỏng weights đã học.
> - **Step 2 (Fine-tune):** Mở khóa toàn bộ mạng, huấn luyện end-to-end với learning rate rất nhỏ. Cosine Annealing giảm LR mượt mà, Early Stopping ngăn overfitting.
> - **Focal Loss:** Giải quyết mất cân bằng dữ liệu — các lớp hiếm (MEL, DF) được tăng trọng số, lớp phổ biến (NV) bị giảm trọng số.

---

## 5. Luồng Đánh giá & Explainable AI (Stage 4)

```mermaid
flowchart TB
    CKPT["models/checkpoints/<br/>best_model.pt"] --> LOAD["Load Model<br/>+ Test Dataset"]
    
    LOAD --> PREDICT["Inference trên Test Set<br/>(y_pred, y_prob)"]
    
    PREDICT --> METRICS["📊 Tính Metrics"]
    PREDICT --> CALIBRATE["🌡️ Temperature Scaling"]
    PREDICT --> GRADCAM["🔥 Grad-CAM++"]
    
    subgraph METRICS_BOX["Metrics đầu ra"]
        M1["AUC-ROC Macro"]
        M2["Balanced Accuracy"]
        M3["Sensitivity / Specificity"]
        M4["F1-weighted"]
        M5["Cohen's Kappa"]
        M6["NPV (Neg. Predictive Value)"]
    end
    
    subgraph CALIBRATE_BOX["Hiệu chuẩn Xác suất"]
        C1["Học tham số T<br/>trên Validation Set"]
        C2["Tính ECE<br/>(Expected Calibration Error)"]
        C3["Vẽ Reliability Diagram"]
    end
    
    subgraph GRADCAM_BOX["Giải thích Mô hình"]
        G1["Hook vào last conv layer"]
        G2["Tính gradient-weighted<br/>activation map"]
        G3["Overlay heatmap<br/>lên ảnh gốc"]
        G4["Tính IoU với<br/>ground-truth mask"]
    end
    
    METRICS --> METRICS_BOX
    CALIBRATE --> CALIBRATE_BOX
    GRADCAM --> GRADCAM_BOX
    
    METRICS_BOX --> OUT1["outputs/logs/<br/>stage_b_metrics.json"]
    CALIBRATE_BOX --> OUT2["outputs/figures/<br/>reliability_diagram.png"]
    GRADCAM_BOX --> OUT3["outputs/xai_heatmaps/<br/>heatmap_*.png"]
    
    METRICS --> CM["outputs/figures/<br/>confusion_matrix.png"]
    METRICS --> ROC["outputs/figures/<br/>roc_curve.png"]

    style METRICS_BOX fill:#1b4332,stroke:#2d6a4f,color:#fff
    style CALIBRATE_BOX fill:#2d1b69,stroke:#533483,color:#fff
    style GRADCAM_BOX fill:#7b2d26,stroke:#e94560,color:#fff
```

> **Giải thích từng thành phần:**
> - **Metrics:** Đánh giá toàn diện mô hình — AUC-ROC cho khả năng phân biệt, Sensitivity cho độ nhạy phát hiện bệnh nguy hiểm, NPV cho tỷ lệ "âm tính thật" (rất quan trọng trong y khoa).
> - **Temperature Scaling:** Hiệu chuẩn xác suất đầu ra để mô hình "tự tin đúng mức" — nếu mô hình nói 80% là melanoma thì thực tế phải đúng ~80% (ECE < 0.05).
> - **Grad-CAM++:** Tạo heatmap hiển thị vùng ảnh nào mô hình "nhìn vào" khi đưa ra quyết định — giúp bác sĩ tin tưởng và kiểm chứng kết quả AI.

---

## 6. Mapping Thư mục ↔ Giai đoạn

```mermaid
flowchart LR
    subgraph FILES["📁 Cấu trúc Thư mục"]
        direction TB
        CF["configs/<br/>base_config.yaml<br/>stage1~4_*.yaml"]
        DR["data/raw/"]
        DP["data/processed/"]
        DC["data/clinical/"]
        SM["src/data/"]
        SMO["src/models/"]
        SX["src/xai/"]
        SP["src/pipelines/"]
        MC["models/checkpoints/"]
        OF["outputs/figures/"]
        OX["outputs/xai_heatmaps/"]
    end

    subgraph STAGES["🔄 Giai đoạn"]
        direction TB
        ST1["Stage 1<br/>Tiền xử lý"]
        ST2["Stage 2<br/>Cân bằng"]
        ST3["Stage 3<br/>Huấn luyện"]
        ST4["Stage 4<br/>Đánh giá & XAI"]
    end

    CF --> ST1
    CF --> ST2
    CF --> ST3
    CF --> ST4
    DR --> ST1
    SM --> ST1
    DP --> ST2
    DP --> ST3
    DC --> ST3
    SMO --> ST3
    SP --> ST3
    MC --> ST4
    SX --> ST4
    ST1 --> DP
    ST1 --> DC
    ST3 --> MC
    ST4 --> OF
    ST4 --> OX
```

---

## 7. Lệnh chạy từng Giai đoạn

| Giai đoạn | Lệnh CLI | Mục đích |
|---|---|---|
| **Setup** | `python check_gpu.py` | Kiểm tra môi trường, GPU, dependencies |
| **Stage 1** | `python -m src.pipelines.run_stage1_preprocess` | Tiền xử lý toàn bộ ảnh thô → ảnh sạch |
| **Stage 3** | `python -m src.pipelines.run_stage3_train` | Huấn luyện mô hình (Stage A + B) |
| **Stage 4** | `python -m src.pipelines.run_stage4_eval_xai` | Đánh giá metric + xuất heatmap Grad-CAM++ |
| **Export** | `python -m src.pipelines.run_export_onnx` | Chuyển đổi sang ONNX để deploy |
| **Test** | `pytest tests/ -v` | Chạy unit tests kiểm tra code |
| **DVC** | `dvc repro` | Chạy lại toàn bộ pipeline tự động (reproducible) |

---

> [!TIP]
> Khi chạy trên **Kaggle Notebooks**, bạn chỉ cần upload thư mục `src/` và `configs/` lên, mount dataset ISIC 2019, rồi chạy các lệnh trên. GPU T4 miễn phí đủ để train trong ~2-4 giờ.
