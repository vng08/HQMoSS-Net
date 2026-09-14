"""
Bộ chạy huấn luyện + kiểm định chéo K-fold cho mọi mô hình trên mọi bộ dữ liệu.

Cách dùng (chạy trong thư mục src/):

    # Mô hình đầy đủ HQMoSS-Net trên GlaS
    python train.py --model HQMoSS_Net --dataset glas

    # Ablation số lớp mạch lượng tử của QuMoE trên PH2
    python train.py --model QuMoE --dataset ph2 --expert-layers 3 3 3 --result-name QuMoE333

    # Đối chứng cổ điển của QuMoE(5,5,5) (mục 5.6.2)
    python train.py --model CMoE --dataset ph2 --expert-layers 5 5 5 --result-name CMoE555

    # Baseline UNet++ trên PH2
    python train.py --model unet_plus --dataset ph2

Kết quả (predictions_fold*.npy, loss_fold*.npy) được lưu vào
    <RESULT_ROOT>/<GlaS|PH2>/<result-name>/
với result-name mặc định là tên mô hình.
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # phải đặt trước khi import tensorflow

import argparse
import warnings

import numpy as np

import config
from data import load_fold_data
from metrics import compute_iou
from models.segmentation import build_variant, VARIANTS
from models.baselines import build_baseline, BASELINE_MODELS

try:
    from silence_tensorflow import silence_tensorflow
    silence_tensorflow()
except ImportError:
    pass

warnings.filterwarnings("ignore", message="You are casting an input of type complex128 *")


def build_model(model_name, expert_layers):
    """Dựng mô hình theo tên: baseline (thư viện) hoặc biến thể tự custom."""
    input_shape = (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3)
    if model_name in BASELINE_MODELS:
        return build_baseline(model_name, input_shape, config.BASELINE_NUM_FILTERS,
                              iou_fn=compute_iou, learning_rate=config.LEARNING_RATE)
    if model_name in VARIANTS:
        return build_variant(model_name, input_shape, config.NUM_FILTERS,
                            expert_layers=expert_layers, learning_rate=config.LEARNING_RATE,
                            iou_fn=compute_iou)
    raise ValueError(
        f"Mô hình không hợp lệ: {model_name}.\n"
        f"  Biến thể tự custom: {list(VARIANTS)}\n"
        f"  Baseline thư viện : {list(BASELINE_MODELS)}"
    )


def execute_fold(model, dataset, fold, image_dir, mask_dir, fold_dir, kind, epochs, batch_size):
    """Huấn luyện + dự đoán cho một fold, trả về (iou, loss, predictions)."""
    X_train, X_val, y_train, y_val = load_fold_data(
        fold, image_dir, mask_dir, fold_dir, config.IMAGE_SIZE, kind=kind
    )

    print("=" * 100)
    print(f"Đã nạp fold {fold} | Train: {X_train.shape} | Val: {X_val.shape}")
    print("=" * 100)

    # Một số baseline (UNet++ deep supervision) có nhiều đầu ra -> nhân bản nhãn.
    multi_output = isinstance(model.output, list)
    y_train_fit = [y_train] * len(model.outputs) if multi_output else y_train
    y_val_fit = [y_val] * len(model.outputs) if multi_output else y_val

    history = model.fit(X_train, y_train_fit, validation_data=(X_val, y_val_fit),
                        epochs=epochs, batch_size=batch_size, verbose=2)

    y_pred = model.predict(X_val)
    if isinstance(y_pred, list):
        y_pred = y_pred[-1]

    iou = compute_iou(y_pred, y_val)
    loss = history.history["loss"]
    return iou, loss, y_pred


def cross_validate(model_name, dataset, expert_layers, epochs, k_folds,
                   result_name, save_weights=False):
    """Chạy K-fold, lưu kết quả numpy mỗi fold, trả về IoU trung bình."""
    image_dir, mask_dir, fold_dir, kind = config.dataset_paths(dataset)
    out_dir = config.result_dir(dataset, result_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Kết quả -> {out_dir}")

    iou_scores = []
    for fold in range(1, k_folds + 1):
        model = build_model(model_name, expert_layers)
        iou, loss, predictions = execute_fold(
            model, dataset, fold, image_dir, mask_dir, fold_dir, kind,
            epochs, config.BATCH_SIZE,
        )
        print(f"{model_name} - Fold {fold} - IoU {float(iou):.4f} - Final Loss {loss[-1]:.4f}")
        iou_scores.append(float(iou))

        np.save(os.path.join(out_dir, f"predictions_fold{fold}.npy"), predictions)
        np.save(os.path.join(out_dir, f"loss_fold{fold}.npy"), np.array(loss))
        if save_weights:
            model.save_weights(os.path.join(out_dir, f"weights_fold{fold}.weights.h5"))

    avg_iou = float(np.mean(iou_scores))
    np.save(os.path.join(out_dir, "result.npy"),
            {"model": result_name, "avg_iou": avg_iou,
             **{f"fold_{i}_iou": s for i, s in enumerate(iou_scores, 1)}})
    return avg_iou


def parse_args():
    p = argparse.ArgumentParser(description="Huấn luyện K-fold cho HQMoSS-Net và các mô hình so sánh.")
    p.add_argument("--model", required=True,
                   help="Tên mô hình. Biến thể: " + ", ".join(VARIANTS)
                        + " | Baseline: " + ", ".join(BASELINE_MODELS))
    p.add_argument("--dataset", required=True, choices=["glas", "ph2"], help="Bộ dữ liệu.")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--folds", type=int, default=config.K_FOLDS)
    p.add_argument("--expert-layers", type=int, nargs=3, default=list(config.DEFAULT_EXPERT_LAYERS),
                   metavar=("L1", "L2", "L3"),
                   help="Số lớp mạch của 3 expert Quantum MoE (chỉ áp dụng khi mô hình có QMoE).")
    p.add_argument("--result-name", default=None,
                   help="Tên thư mục kết quả (mặc định = tên mô hình). VD: QuMoE333 cho ablation.")
    p.add_argument("--save-weights", action="store_true", help="Lưu trọng số mỗi fold (cần cho Seg-Grad-CAM).")
    return p.parse_args()


def main():
    args = parse_args()
    result_name = args.result_name or args.model
    print("\n" + "=" * 100)
    print(f"Huấn luyện {args.model} trên {args.dataset.upper()} ({args.folds} folds, {args.epochs} epochs)")
    print("=" * 100)

    avg_iou = cross_validate(
        model_name=args.model, dataset=args.dataset,
        expert_layers=tuple(args.expert_layers), epochs=args.epochs,
        k_folds=args.folds, result_name=result_name, save_weights=args.save_weights,
    )

    print("=" * 100)
    print(f"IoU trung bình ({args.model} / {args.dataset.upper()}): {avg_iou:.4f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
