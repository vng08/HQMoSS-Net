# HQMoSS-Net

**HQMoSS-Net** — Mạng phân vùng ảnh y khoa lai lượng tử (Quantum-MoE) với khối Selective Scan, đa tỉ lệ và tầm xa.

---

## Mục lục

- [1. Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
- [2. Cài đặt](#2-cài-đặt)
- [3. Cấu trúc gói code](#3-cấu-trúc-gói-code)
- [4. Chuẩn bị dữ liệu & kết quả](#4-chuẩn-bị-dữ-liệu--kết-quả)
- [5. Kiểm tra cài đặt](#5-kiểm-tra-cài-đặt)
- [6. Các mô hình khả dụng](#6-các-mô-hình-khả-dụng)
- [7. Huấn luyện (train.py)](#7-huấn-luyện-trainpy)
- [8. Đánh giá & báo cáo (evaluate.py)](#8-đánh-giá--báo-cáo-evaluatepy)
- [9. Giải thích mô hình — Seg-Grad-CAM & độ đo (explain.py)](#9-giải-thích-mô-hình--seg-grad-cam--độ-đo-explainpy)
- [10. Quy trình tái lập toàn bộ kết quả](#10-quy-trình-tái-lập-toàn-bộ-kết-quả)

---

## 1. Yêu cầu hệ thống

- Python 3.10 hoặc 3.11 (khuyến nghị; PennyLane 0.41 chưa hỗ trợ 3.13+).
- RAM tối thiểu 8 GB (mô phỏng lượng tử tốn bộ nhớ). Có GPU sẽ nhanh hơn nhưng không bắt buộc.
- Hệ điều hành: Windows / Linux / macOS.

## 2. Cài đặt

### 2.1. Tạo môi trường ảo (khuyến nghị)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2.2. Cài thư viện

Trong thư mục `src/`, chạy:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Ghi chú:**
- Các phiên bản trong `requirements.txt` đã được cố định đúng như môi trường Kaggle nơi thí nghiệm được thực hiện, nhằm tái lập chính xác kết quả.
- Nếu chỉ chạy các mô hình baseline (UNet++, Attention U-Net, R2U-Net, VNet) thì không bắt buộc cài nhóm PennyLane, nhưng nên cài đủ để chạy được HQMoSS-Net và phần trực quan Seg-Grad-CAM.
- Phần độ đo interpretability (`explain.py --mode metrics/compare`) cần thêm `scipy`; gói này đã có sẵn trong `requirements.txt` nên không phải cài riêng.

## 3. Cấu trúc gói code

```
25D2-KL-KHMT23/
├── src/
│   ├── config.py            Cấu hình đường dẫn + siêu tham số
│   ├── data.py              Nạp & tiền xử lý ảnh/mask cho GlaS và PH2
│   ├── metrics.py           IoU (khi train) và bộ chỉ số đầy đủ (khi báo cáo)
│   ├── train.py             Huấn luyện + kiểm định chéo K-fold (dòng lệnh)
│   ├── evaluate.py          Tính bảng chỉ số báo cáo + vẽ biểu đồ so sánh
│   ├── explain.py           Seg-Grad-CAM + độ đo interpretability (4 chế độ)
│   ├── xai_metrics.py       Độ đo định lượng chất lượng CAM + lọc nhiễu component
│   ├── router_analysis.py   Cân bằng định tuyến QuantumMoE + kiểm định Mann-Whitney U
│   ├── requirements.txt     Danh sách thư viện
│   ├── models/
│   │   ├── blocks.py        RDMS, khối conv encoder/decoder, SelectiveScan, LSS
│   │   ├── quantum.py       Quantum MoE (PQC, expert, router) + QuFeX (QCNN)
│   │   ├── classical_moe.py Classical MoE (CMoE) -- đối chứng cổ điển của QuMoE
│   │   ├── segmentation.py  Bộ dựng mô hình hợp nhất + các biến thể ablation
│   │   └── baselines.py     Các baseline U-Net (UNet++, Attention, R2U-Net, VNet)
│   ├── Data/                Dữ liệu: GlaS_Fold_Dataset/, PH2_Dataset/
│   ├── Reuslt/              Kết quả numpy: GlaS/, PH2/ (theo từng mô hình)
│   └── XAI-result/          XAI_Ph2_HQMoSS/ (HPweights_fold1.weights.h5 + gradcam/)
├── HuongDanCaiDat.txt
└── HuongDanSuDung.txt
```

> **Lưu ý:** Mọi lệnh trong README này đều chạy **TRONG** thư mục `src/` (`cd 25D2-KL-KHMT23/src` trước).

## 4. Chuẩn bị dữ liệu & kết quả

Dữ liệu và kết quả numpy được đặt **NGAY TRONG** thư mục `src/` theo cấu trúc:

```
25D2-KL-KHMT23/src/
├── Data/
│   ├── GlaS_Fold_Dataset/   <-- ảnh + mask + folds của GlaS 2015
│   │   └── images/  masks/  folds/
│   └── PH2_Dataset/         <-- ảnh + mask + folds của PH2
│       └── images/  masks/  folds/
├── Reuslt/
│   ├── GlaS/                <-- kết quả numpy trên GlaS (theo từng mô hình)
│   └── PH2/                 <-- kết quả numpy trên PH2 (theo từng mô hình)
└── XAI-result/
    └── XAI_Ph2_HQMoSS/      <-- HPweights_fold1.weights.h5 + thư mục gradcam/ (ảnh Seg-Grad-CAM)
```

Mỗi thư mục `folds/` chứa các file `fold_{1..5}_train.csv` và `fold_{1..5}_val.csv` với 2 cột: `image_filename`, `mask_filename`.

### Đặt dữ liệu ở vị trí khác

Nếu muốn đặt dữ liệu ở vị trí khác, có 2 cách trỏ đường dẫn:

**(a) Đặt biến môi trường trước khi chạy:**

Windows:
```powershell
$env:HQMOSS_DATA_ROOT="D:\duong\dan\du\lieu"
$env:HQMOSS_RESULT_ROOT="D:\duong\dan\ket\qua"
$env:HQMOSS_XAI_ROOT="D:\duong\dan\xai"
```

Linux:
```bash
export HQMOSS_DATA_ROOT=/duong/dan/du/lieu
export HQMOSS_RESULT_ROOT=/duong/dan/ket/qua
export HQMOSS_XAI_ROOT=/duong/dan/xai
```

**(b)** Hoặc sửa trực tiếp `DATA_ROOT` / `RESULT_ROOT` / `XAI_ROOT` trong file `src/config.py`.

## 5. Kiểm tra cài đặt

Trong thư mục `src/`, chạy thử:
```bash
python -c "import tensorflow, pennylane, scipy; print('TF', tensorflow.__version__, '| PennyLane', pennylane.__version__, '| SciPy', scipy.__version__)"
```

Nếu in ra phiên bản mà không lỗi là đã cài thành công.

> **Kiểm chứng nhanh không cần huấn luyện lại:** repo này đã có sẵn toàn bộ
> kết quả dùng trong báo cáo — `predictions_fold*.npy`/`loss_fold*.npy` của
> mọi mô hình trong `src/Reuslt/<GlaS|PH2>/<model>/`, và trọng số fold 1 của
> HQMoSS-Net trong `src/XAI-result/XAI_Ph2_HQMoSS/HPweights_fold1.weights.h5`.
> Muốn kiểm chứng lại số liệu/hình ảnh đã báo cáo ở Chương 5 mà **không cần
> huấn luyện lại** (mục 7 tốn nhiều thời gian do phải mô phỏng mạch lượng
> tử), chạy thẳng các lệnh ở mục 8 và 9 bên dưới — `evaluate.py` và
> `explain.py` đều đọc trực tiếp từ các file `.npy`/`.h5` đã có sẵn, KHÔNG
> gọi lại `train.py`. Mục 7 (huấn luyện) chỉ cần thiết khi muốn tạo mới một
> kết quả chưa có (ví dụ thử `--expert-layers` khác) hoặc kiểm tra khả năng
> tái lập khi huấn luyện lại từ đầu (không cố định seed nên có thể lệch nhẹ
> so với báo cáo).

## 6. Các mô hình khả dụng

**a) Họ mô hình tự thiết kế** (8 mô hình ablation — bật/tắt 3 thành phần RDMS / Quantum MoE / LSS), truyền qua `--model`:

| Tên | Thành phần |
|---|---|
| `UNet` | (không thành phần nào) |
| `UNet_LSS` | + LSS |
| `UNet_RDMS` | + RDMS |
| `UNet_RDMS_LSS` | + RDMS + LSS |
| `QuMoE` | + Quantum MoE |
| `QMoE_LSS` | + Quantum MoE + LSS |
| `QMoE_RDMS` | + Quantum MoE + RDMS |
| `HQMoSS_Net` | đầy đủ: RDMS + Quantum MoE + LSS **← mô hình chính** |

**b) Baseline** gọi qua thư viện `keras_unet_collection`:
`unet_plus` (UNet++), `att_unet` (Attention U-Net), `r2_unet` (R2U-Net), `vnet` (VNet 2D)

**c) Baseline lượng tử kiểu QCNN** (QuFeX gốc — mô hình tham chiếu ta lấy ý tưởng):
`QuNet` (conv thường + khối QuFeX 12 qubit ở bottleneck, không MoE/LSS)

**d) Đối chứng cổ điển** (dùng cho ablation số lớp mạch, mục 5.6.2):
`CMoE` (conv thường + Classical MoE ở bottleneck: kiến trúc mirror y hệt QuMoE — cùng 3 patch size 2x2/2x4/8x8, cùng router + residual — nhưng expert là MLP cổ điển thay vì PQC lượng tử. Dùng để cô lập xem lợi ích quan sát được ở QuMoE có thực sự đến từ biểu diễn lượng tử hay chỉ từ cấu trúc router/residual.)

## 7. Huấn luyện (train.py)

**Cú pháp:**
```bash
python train.py --model <TÊN_MÔ_HÌNH> --dataset <glas|ph2> [tuỳ chọn]
```

**Ví dụ:**
```bash
# Mô hình chính HQMoSS-Net trên GlaS và PH2
python train.py --model HQMoSS_Net --dataset glas
python train.py --model HQMoSS_Net --dataset ph2

# Baseline UNet++ trên PH2
python train.py --model unet_plus --dataset ph2

# Ablation từng thành phần (ví dụ chạy UNet + LSS trên PH2)
python train.py --model UNet_LSS --dataset ph2

# Baseline lượng tử QuFeX gốc (đặt tên thư mục kết quả là Qunet_12_1 để khớp báo cáo)
python train.py --model QuNet --dataset ph2 --result-name Qunet_12_1
python train.py --model QuNet --dataset glas --result-name Qunet_12_1
```

**Tuỳ chọn thường dùng:**

| Tuỳ chọn | Ý nghĩa |
|---|---|
| `--epochs N` | Số epoch (mặc định 60) |
| `--folds K` | Số fold (mặc định 5) |
| `--result-name TÊN` | Tên thư mục kết quả (mặc định = tên mô hình) |
| `--save-weights` | Lưu trọng số mỗi fold (**BẮT BUỘC** nếu muốn chạy Seg-Grad-CAM sau) |
| `--expert-layers a b c` | Số lớp mạch của 3 expert Quantum MoE |

**Ablation số lớp mạch lượng tử** (mô hình QuMoE, trên PH2):
```bash
python train.py --model QuMoE --dataset ph2 --expert-layers 1 1 1 --result-name QuMoE111
python train.py --model QuMoE --dataset ph2 --expert-layers 2 2 2 --result-name QuMoE222
python train.py --model QuMoE --dataset ph2 --expert-layers 3 3 3 --result-name QuMoE333
python train.py --model QuMoE --dataset ph2 --expert-layers 4 4 4 --result-name QuMoE444
python train.py --model QuMoE --dataset ph2 --expert-layers 5 5 5 --result-name QuMoE555
```

**Đối chứng cổ điển của QuMoE(5,5,5)** (mục 5.6.2, tên thư mục `CMoE555` để khớp báo cáo):
```bash
python train.py --model CMoE --dataset ph2 --expert-layers 5 5 5 --result-name CMoE555
```

**Kết quả mỗi lần chạy** được lưu vào `src/Reuslt/<GlaS|PH2>/<result-name>/`:
- `predictions_fold{1..5}.npy` — mảng dự đoán trên tập validation
- `loss_fold{1..5}.npy` — đường cong loss huấn luyện
- `result.npy` — IoU trung bình + IoU từng fold
- `weights_fold{1..5}.weights.h5` — nếu bật `--save-weights`

> **Lưu ý:** không cố định seed → mỗi lần chạy khởi tạo trọng số ngẫu nhiên nên kết quả có thể lệch nhẹ giữa các lần.

## 8. Đánh giá & báo cáo (evaluate.py)

Tính bảng chỉ số (mean ± std qua 5 fold) từ các file predictions đã lưu:

```bash
python evaluate.py                     # tất cả các nhóm, lưu CSV vào ../reports
python evaluate.py --plots             # kèm biểu đồ so sánh (.png)
python evaluate.py --group qmoe        # chỉ ablation số lớp mạch (PH2)
python evaluate.py --group components  # chỉ ablation thành phần (PH2)
python evaluate.py --group models --dataset glas   # so sánh mô hình trên GlaS
```

Bảng chỉ số gồm: Soft/Hard IoU, Soft/Hard Dice, Accuracy, Sensitivity, Specificity, Precision. File CSV được ghi vào thư mục `25D2-KL-KHMT23/reports/`.

**Kèm theo (khi có đủ dữ liệu):**
- Nhóm `components`: bảng delta (mức cải thiện Soft IoU so với UNet, %), lưu `components_delta_ph2.csv`.
- Nhóm `models` (khi `--plots`): thêm biểu đồ hiệu quả tham số (số tham số theo trục log vs Soft IoU), lưu `models_efficiency_<ds>.png`.

Mô hình `QuNet` (QuFeX gốc) đã được đưa vào cả nhóm `components` lẫn nhóm `models` làm mốc so sánh (thư mục kết quả tên `Qunet_12_1`).
Nhóm `qmoe` (ablation số lớp mạch) có thêm dòng "CMoE with 5 Experts" (thư mục kết quả `CMoE555`) làm đối chứng cổ điển cho cấu hình (5,5,5) lượng tử.

## 9. Giải thích mô hình — Seg-Grad-CAM & độ đo (explain.py)

Mặc định nạp weights từ `src/XAI-result/XAI_Ph2_HQMoSS/HPweights_fold1.weights.h5`. Layer đích mặc định được chọn theo **vị trí cấu trúc** (output decoder stage áp chót, cùng resolution giữa mọi kiến trúc) nên tiêu chí giải thích nhất quán khi so sánh.

Có 4 chế độ qua `--mode`:

**a) `cam` (mặc định)** — vẽ heatmap Seg-Grad-CAM:
```bash
python explain.py --dataset ph2 --fold 1 --num-samples 5
```
Sinh ảnh (Image | Ground Truth | Prediction | Heatmap) và `cams.npy` trong `src/XAI-result/XAI_Ph2_HQMoSS/gradcam/`.

**b) `metrics`** — tính đầy đủ độ đo định lượng chất lượng CAM trên toàn bộ val set (localization: IoU/Dice theo Otsu & area-matched, energy pointing game; boundary: Boundary IoU, Hausdorff95; faithfulness: Deletion/Insertion AUC):
```bash
python explain.py --mode metrics --dataset ph2 --model HQMoSS_Net --weights <PATH>
```
Kết quả lưu vào `.../metrics/`. Với PH2 (1 tổn thương/ảnh) CAM lọc bằng keep-largest-component; với GlaS (nhiều tuyến/ảnh) dùng remove-small-components — chọn tự động theo `--dataset`.

**c) `compare`** — so sánh Seg-Grad-CAM giữa HQMoSS-Net và UNet, chọn các ảnh HQMoSS định vị tốt hơn UNet nhiều nhất:
```bash
python explain.py --mode compare --dataset ph2 \
       --weights <HQ_WEIGHTS> --unet-weights <UNET_WEIGHTS>
```
Kết quả lưu vào `src/XAI-result/comparison_<dataset>/`.

**d) `router`** — phân tích cân bằng định tuyến (routing load balance) của khối QuantumMoE trên **một** checkpoint (forward-pass, không huấn luyện lại): tỷ trọng trung bình mỗi nhánh, entropy định tuyến, imbalance range, CV:
```bash
python explain.py --mode router --dataset ph2 --fold 1 --weights <PATH>
```
Lưu bảng theo từng ảnh vào `.../router/router_balance_<dataset>_fold<k>_per_image.csv`.

Để so sánh cân bằng định tuyến giữa hai tập dữ liệu bằng kiểm định Mann-Whitney U (kèm cỡ hiệu ứng rank-biserial r), chạy mode `router` cho tập thứ hai với `--compare-csv` trỏ tới CSV vừa tạo ở tập đầu:
```bash
python explain.py --mode router --dataset glas --fold 1 --weights <PATH> \
       --compare-csv <ph2_per_image.csv> --compare-label PH2
```
Xem công thức ở `src/router_analysis.py` và cơ sở lý thuyết ở khóa luận.

**Tuỳ chọn thường dùng:**

| Tuỳ chọn | Ý nghĩa |
|---|---|
| `--model <TÊN>` | mô hình để giải thích ở chế độ cam/metrics (mặc định `HQMoSS_Net`; có thể dùng `UNet`) |
| `--weights <đường_dẫn>` | trỏ tới file weights (VD `weights_fold*.weights.h5` sinh ra khi train với `--save-weights`) |
| `--layer <tên_layer>` | chỉ định layer đích thủ công (mặc định: chọn tự động) |
| `--stages-from-end N` | lấy output decoder stage thứ N tính từ cuối (mặc định 2) |

## 10. Quy trình tái lập toàn bộ kết quả

> Quy trình dưới đây dành cho việc **tạo mới** kết quả từ đầu (ví dụ trên một
> máy chưa có gì trong `Reuslt/`). Nếu chỉ muốn **kiểm chứng lại báo cáo**, bỏ
> qua bước 1-3 (huấn luyện) và chạy thẳng bước 4-5 — xem khung "Kiểm chứng
> nhanh không cần huấn luyện lại" ở mục 5.

Gợi ý thứ tự:

1. Huấn luyện `HQMoSS_Net`, các baseline và `QuNet` trên cả `glas` & `ph2` (mục 7).
2. Huấn luyện các biến thể ablation thành phần trên `ph2`:
   `UNet`, `UNet_LSS`, `UNet_RDMS`, `UNet_RDMS_LSS`, `QMoE_LSS`, `QMoE_RDMS`
   (dùng `QuMoE555` làm cấu hình QuMoE trong bảng ablation).
3. Huấn luyện ablation số lớp mạch `QuMoE111`..`QuMoE555` trên `ph2`, cùng đối
   chứng cổ điển `CMoE555` (mục 7 — `CMoE` với `--expert-layers 5 5 5`).
4. Chạy `evaluate.py --plots` để sinh bảng CSV + biểu đồ (kèm bảng delta và biểu đồ hiệu quả tham số).
5. *(Tuỳ chọn)* Chạy `explain.py` (mode `cam`/`metrics`/`compare`/`router`) để có ảnh Seg-Grad-CAM, bảng độ đo interpretability và phân tích cân bằng định tuyến.

---

*Xem thêm `HuongDanCaiDat.txt` và `HuongDanSuDung.txt` để biết chi tiết gốc.*