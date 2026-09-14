"""
Cấu hình trung tâm cho toàn bộ thí nghiệm HQMoSS-Net.

Mọi đường dẫn dữ liệu / kết quả và siêu tham số huấn luyện đều tập trung ở đây
để các module khác (data, train, evaluate, explain) dùng chung, tránh lặp lại.

Cấu trúc mặc định giả định dữ liệu và kết quả nằm NGAY TRONG thư mục src/:
    src/
    ├── Data/          -> GlaS_Fold_Dataset/, PH2_Dataset/
    ├── Reuslt/        -> GlaS/, PH2/   (kết quả numpy theo từng mô hình)
    └── XAI-result/    -> XAI_Ph2_HQMoSS/ (weights + ảnh Seg-Grad-CAM)

Muốn đổi vị trí: sửa DATA_ROOT / RESULT_ROOT / XAI_ROOT bên dưới, hoặc đặt biến
môi trường HQMOSS_DATA_ROOT / HQMOSS_RESULT_ROOT / HQMOSS_XAI_ROOT trước khi chạy.
"""

import os

# ---------------------------------------------------------------------------
# 1. Đường dẫn gốc (dữ liệu và kết quả nằm trong chính thư mục src/)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # = .../25D2-KL-KHMT23/src

DATA_ROOT = os.environ.get("HQMOSS_DATA_ROOT", os.path.join(_THIS_DIR, "Data"))
RESULT_ROOT = os.environ.get("HQMOSS_RESULT_ROOT", os.path.join(_THIS_DIR, "Reuslt"))
XAI_ROOT = os.environ.get("HQMOSS_XAI_ROOT", os.path.join(_THIS_DIR, "XAI-result"))

# Thư mục kết quả Seg-Grad-CAM (chứa HPweights_fold1.weights.h5 và gradcam/).
XAI_RESULT_DIR = os.path.join(XAI_ROOT, "XAI_Ph2_HQMoSS")

# ---------------------------------------------------------------------------
# 2. Mô tả 2 bộ dữ liệu (GlaS 2015 và PH2)
# ---------------------------------------------------------------------------
# - kind: "glas" hoặc "ph2" -> quyết định cách ghép tên mask và cách nhị phân hoá.
#     glas: mask = <tên ảnh>_anno.<ext>, nhị phân bằng (mask > 0)
#     ph2 : mask = <tên ảnh>,            nhị phân bằng (mask/255 > 0.5)
DATASETS = {
    "glas": {
        "kind": "glas",
        "base_dir": os.path.join(DATA_ROOT, "GlaS_Fold_Dataset"),
        "result_dir": os.path.join(RESULT_ROOT, "GlaS"),
    },
    "ph2": {
        "kind": "ph2",
        "base_dir": os.path.join(DATA_ROOT, "PH2_Dataset"),
        "result_dir": os.path.join(RESULT_ROOT, "PH2"),
    },
}


def dataset_paths(dataset):
    """Trả về (image_dir, mask_dir, fold_dir, kind) cho một bộ dữ liệu."""
    cfg = DATASETS[dataset.lower()]
    base = cfg["base_dir"]
    return (
        os.path.join(base, "images"),
        os.path.join(base, "masks"),
        os.path.join(base, "folds"),
        cfg["kind"],
    )


def result_dir(dataset, model_name):
    """Thư mục lưu kết quả numpy: <RESULT_ROOT>/<GlaS|PH2>/<model_name>."""
    return os.path.join(DATASETS[dataset.lower()]["result_dir"], model_name)


# ---------------------------------------------------------------------------
# 3. Siêu tham số huấn luyện dùng chung
# ---------------------------------------------------------------------------
IMAGE_SIZE = (192, 256)          # (H, W) sau khi resize
BATCH_SIZE = 18
EPOCHS = 60
K_FOLDS = 5
NUM_FILTERS = [8, 16, 32, 16, 8]  # số filter mỗi tầng encoder của họ mô hình tự custom
LEARNING_RATE = 0.001

# Số filter cho các mô hình baseline gọi qua thư viện keras_unet_collection.
BASELINE_NUM_FILTERS = [12, 24, 48, 64, 128]

# Cấu hình số lớp mạch lượng tử cho mỗi expert của Quantum MoE (mặc định).
DEFAULT_EXPERT_LAYERS = (5, 5, 5)
