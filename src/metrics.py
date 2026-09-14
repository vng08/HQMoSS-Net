"""
Các độ đo đánh giá phân vùng (segmentation metrics).

- compute_iou: IoU "mềm" tính trực tiếp trên xác suất, dùng làm metric trong lúc
  huấn luyện (đúng như trong các notebook gốc).
- compute_metrics: bộ độ đo đầy đủ (soft/hard IoU & Dice, accuracy, sensitivity,
  specificity, precision) dùng cho báo cáo cuối cùng (evaluate.py).
"""

import numpy as np
import tensorflow as tf

EPS = 1e-7


def compute_iou(y_pred, y_true):
    """IoU mềm trên cả batch, dùng làm Keras metric khi train."""
    epsilon = tf.keras.backend.epsilon()
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    return (intersection + epsilon) / (union + epsilon)


def iou_one(pred, true):
    """IoU của một ảnh (giá trị float), tiện cho việc tìm ca dự đoán tệ nhất."""
    return float(compute_iou(pred, true).numpy())


def compute_metrics(y_true, y_pred, threshold=0.5):
    """
    Tính đầy đủ các độ đo cho một tập dự đoán.

    y_true, y_pred: mảng numpy (N, H, W) hoặc (N, H, W, 1). y_pred là xác suất [0, 1].
    Trả về dict gồm soft/hard IoU & Dice và các chỉ số phân loại pixel.
    """
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    if y_true.ndim == 3:
        y_true = np.expand_dims(y_true, axis=-1)
    if y_pred.ndim == 3:
        y_pred = np.expand_dims(y_pred, axis=-1)

    y_pred = np.clip(y_pred, 0, 1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)

    # --- Độ đo "mềm" (trực tiếp trên xác suất) ---
    soft_intersection = np.sum(y_true_f * y_pred_f)
    soft_union = np.sum(y_true_f) + np.sum(y_pred_f) - soft_intersection
    soft_sum = np.sum(y_true_f) + np.sum(y_pred_f)

    soft_iou = (soft_intersection + EPS) / (soft_union + EPS)
    soft_dice = (2 * soft_intersection + EPS) / (soft_sum + EPS)

    # --- Độ đo "cứng" (sau khi ngưỡng hoá) ---
    y_pred_hard = (y_pred >= threshold).astype(np.float32).reshape(-1)
    y_true_hard = y_true_f

    tp = np.sum((y_true_hard == 1) & (y_pred_hard == 1))
    tn = np.sum((y_true_hard == 0) & (y_pred_hard == 0))
    fp = np.sum((y_true_hard == 0) & (y_pred_hard == 1))
    fn = np.sum((y_true_hard == 1) & (y_pred_hard == 0))

    hard_dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
    hard_iou = (tp + EPS) / (tp + fp + fn + EPS)
    accuracy = (tp + tn + EPS) / (tp + tn + fp + fn + EPS)
    sensitivity = (tp + EPS) / (tp + fn + EPS)
    specificity = (tn + EPS) / (tn + fp + EPS)
    precision = (tp + EPS) / (tp + fp + EPS)

    return {
        "soft_dice": soft_dice,
        "soft_iou": soft_iou,
        "hard_dice": hard_dice,
        "hard_iou": hard_iou,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
    }
