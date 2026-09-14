"""
Nạp và tiền xử lý dữ liệu ảnh y khoa cho 2 bộ GlaS 2015 và PH2.

Điểm khác nhau giữa 2 bộ (được xử lý thống nhất qua tham số ``kind``):
  * glas: tên mask = <tên ảnh>_anno.<ext>; nhãn nhị phân bằng (mask > 0);
           đọc mask với interpolation="nearest" để giữ đúng biên nhãn.
  * ph2 : tên mask = trùng tên ảnh; chuẩn hoá /255 rồi ngưỡng (mask > 0.5).

Việc chia fold dựa trên các file CSV có sẵn trong <base>/folds/fold_{k}_{train,val}.csv
với 2 cột: image_filename, mask_filename.
"""

import os

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image


def _mask_name(img_filename, kind):
    """Suy ra tên file mask từ tên file ảnh theo quy ước của từng bộ dữ liệu."""
    name, ext = os.path.splitext(img_filename)
    if kind == "glas":
        return name + "_anno" + ext
    if kind == "ph2":
        return img_filename
    raise ValueError(f"Unknown dataset kind: {kind}")


def load_images_and_masks(image_dir, mask_dir, image_size=(192, 256), kind="ph2"):
    """
    Đọc toàn bộ ảnh + mask trong thư mục, trả về:
        images (N, H, W, 3) đã chuẩn hoá [0, 1],
        masks  (N, H, W, 1) nhị phân {0, 1},
        image_filenames, mask_filenames (danh sách tên tương ứng).
    """
    image_filenames = sorted(f for f in os.listdir(image_dir) if not f.startswith("."))

    images, masks, mask_filenames = [], [], []

    for img_filename in image_filenames:
        mask_filename = _mask_name(img_filename, kind)
        img_path = os.path.join(image_dir, img_filename)
        mask_path = os.path.join(mask_dir, mask_filename)

        if not os.path.exists(mask_path):
            print(f"Missing mask: {mask_path}")
            continue

        img = image.load_img(img_path, target_size=image_size)
        img = image.img_to_array(img) / 255.0

        mask = image.load_img(
            mask_path, target_size=image_size,
            color_mode="grayscale", interpolation="nearest",
        )
        mask = image.img_to_array(mask)

        if kind == "ph2":
            mask = (mask / 255.0 > 0.5).astype("float32")
        else:  # glas
            mask = (mask > 0).astype("float32")

        images.append(img)
        masks.append(mask)
        mask_filenames.append(mask_filename)

    return np.array(images), np.array(masks), image_filenames, mask_filenames


def load_fold_data(fold, image_dir, mask_dir, fold_dir, image_size=(192, 256), kind="ph2"):
    """
    Nạp dữ liệu của một fold, trả về (X_train, X_val, y_train, y_val).
    Danh sách file train/val lấy từ fold_{fold}_train.csv và fold_{fold}_val.csv.
    """
    train_df = pd.read_csv(os.path.join(fold_dir, f"fold_{fold}_train.csv"))
    val_df = pd.read_csv(os.path.join(fold_dir, f"fold_{fold}_val.csv"))

    train_img_files = set(train_df["image_filename"].values)
    val_img_files = set(val_df["image_filename"].values)
    train_mask_files = set(train_df["mask_filename"].values)
    val_mask_files = set(val_df["mask_filename"].values)

    images, masks, image_filenames, mask_filenames = load_images_and_masks(
        image_dir, mask_dir, image_size=image_size, kind=kind
    )

    train_images = [images[i] for i in range(len(image_filenames)) if image_filenames[i] in train_img_files]
    val_images = [images[i] for i in range(len(image_filenames)) if image_filenames[i] in val_img_files]
    train_masks = [masks[i] for i in range(len(mask_filenames)) if mask_filenames[i] in train_mask_files]
    val_masks = [masks[i] for i in range(len(mask_filenames)) if mask_filenames[i] in val_mask_files]

    return (
        np.array(train_images), np.array(val_images),
        np.array(train_masks), np.array(val_masks),
    )
