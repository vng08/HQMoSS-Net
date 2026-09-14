"""
Các mô hình baseline U-Net gọi qua thư viện keras_unet_collection, dùng để so
sánh với HQMoSS-Net (UNet++, Attention U-Net, R2U-Net, VNet 2D, ResU-Net, ...).

Bảng ánh xạ tên hiển thị trong báo cáo  ->  khoá model ở đây:
    UNet++         -> unet_plus
    Attention UNet -> att_unet
    R2U-Net        -> r2_unet
    VNet (2D)      -> vnet
"""

import tensorflow as tf
from keras_unet_collection import models

from metrics import compute_iou

# Các mô hình baseline được dùng trong khoá luận (khoá -> tên hiển thị / thư mục kết quả).
BASELINE_MODELS = {
    "unet_plus": "UNetPlusPlus",
    "att_unet": "AttentionUNet",
    "r2_unet": "R2UNet",
    "vnet": "VNet",
}


def build_baseline(model_name, input_shape, num_filters, iou_fn=compute_iou,
                   learning_rate=0.001, deep_supervision=False):
    """Dựng và compile một mô hình baseline từ keras_unet_collection."""
    common = dict(
        input_size=input_shape, n_labels=1, activation="ReLU",
        output_activation="Sigmoid", batch_norm=True, pool=True, unpool=False,
    )

    if model_name == "unet":
        model = models.unet_2d(**common, filter_num=num_filters,
                               stack_num_down=2, stack_num_up=2, name="unet")

    elif model_name == "unet_plus":
        model = models.unet_plus_2d(**common, filter_num=num_filters, stack_num_down=2,
                                    stack_num_up=2, deep_supervision=deep_supervision, name="unet_plus")

    elif model_name == "unet_3plus":
        model = models.unet_3plus_2d(**common, filter_num_down=num_filters, filter_num_skip="auto",
                                     filter_num_aggregate="auto", stack_num_down=2, stack_num_up=2,
                                     deep_supervision=False, name="unet_3plus")

    elif model_name == "att_unet":
        model = models.att_unet_2d(**common, filter_num=num_filters, stack_num_down=2,
                                   stack_num_up=2, atten_activation="ReLU", attention="add", name="att_unet")

    elif model_name == "r2_unet":
        model = models.r2_unet_2d(**common, filter_num=num_filters, stack_num_down=2,
                                  stack_num_up=2, recur_num=2, name="r2_unet")

    elif model_name == "resunet":
        model = models.resunet_a_2d(**common, filter_num=num_filters, dilation_num=[1, 3, 15, 31],
                                    aspp_num_down=num_filters[-1], aspp_num_up=num_filters[0], name="resunet")

    elif model_name == "u2net":
        model = models.u2net_2d(**common, filter_num_down=num_filters, filter_num_up=num_filters[::-1],
                                filter_mid_num_down="auto", filter_mid_num_up="auto",
                                filter_4f_num="auto", filter_4f_mid_num="auto", name="u2net")

    elif model_name == "vnet":
        model = models.vnet_2d(**common, filter_num=num_filters, res_num_ini=1, res_num_max=3, name="vnet")

    elif model_name == "swin_unet":
        model = models.swin_unet_2d(
            input_size=input_shape, filter_num_begin=num_filters[0], n_labels=1, depth=4,
            stack_num_down=2, stack_num_up=2, patch_size=(4, 4), num_heads=[4, 8, 8, 8],
            window_size=[4, 4, 4, 4], num_mlp=512, output_activation="Sigmoid",
            shift_window=True, name="swin_unet",
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    if model_name == "unet_plus" and deep_supervision:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=["binary_crossentropy"] * len(model.outputs))
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="binary_crossentropy", metrics=[iou_fn])

    return model
