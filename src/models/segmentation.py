"""
Bộ dựng mô hình phân vùng hợp nhất cho họ mô hình tự thiết kế của khoá luận.

Toàn bộ 8 biến thể trong nghiên cứu ablation đều là các cấu hình bật/tắt của
cùng một khung encoder–bottleneck–decoder, điều khiển bởi 3 công tắc:

    use_rdms  : dùng khối RDMS thay cho conv thường ở encoder/decoder
    use_qmoe  : dùng Quantum MoE ở bottleneck thay cho 2 lớp conv
    use_qufex : dùng khối QuFeX (QCNN) ở bottleneck (baseline lượng tử so sánh)
    use_cmoe  : dùng Classical MoE ở bottleneck (đối chứng cổ điển của QuMoE)
    use_lss   : chèn khối LSS (selective scan) ngay sau bottleneck

Bảng ánh xạ biến thể  <->  công tắc:

    UNet              : rdms=F, qmoe=F, lss=F
    UNet_LSS          : rdms=F, qmoe=F, lss=T
    UNet_RDMS         : rdms=T, qmoe=F, lss=F
    UNet_RDMS_LSS     : rdms=T, qmoe=F, lss=T
    QuMoE             : rdms=F, qmoe=T, lss=F
    QMoE_LSS          : rdms=F, qmoe=T, lss=T
    QMoE_RDMS         : rdms=T, qmoe=T, lss=F
    HQMoSS_Net (đầy đủ): rdms=T, qmoe=T, lss=T
    QuNet             : rdms=F, qufex=T (bottleneck QCNN, không MoE/LSS)
    CMoE              : rdms=F, cmoe=T (đối chứng cổ điển của QuMoE, dùng cho
                        ablation số lớp mạch ở mục 5.6.2)
"""

import tensorflow as tf
import tensorflow.keras.layers as layers

from metrics import compute_iou
from models.blocks import conv_block, convT_block, LSS
from models.quantum import QuantumMoE, QuFeX
from models.classical_moe import ClassicalMoE


def build_segmentation_model(
    input_shape,
    num_filters,
    use_rdms=False,
    use_qmoe=False,
    use_lss=False,
    use_qufex=False,
    use_cmoe=False,
    expert_layers=(5, 5, 5),
    learning_rate=0.001,
    iou_fn=compute_iou,
    name=None,
):
    """Dựng một mô hình phân vùng theo cấu hình các công tắc rdms/qmoe/qufex/cmoe/lss."""
    inputs = layers.Input(input_shape)
    skip_conn = []
    temp = inputs

    # --- Encoder ---
    for f in num_filters:
        c, p = conv_block(temp, f, use_rdms=use_rdms)
        skip_conn.append(c)
        temp = p

    # --- Bottleneck ---
    if use_qmoe:
        temp, _ = QuantumMoE(temp, expert_layers=expert_layers, name="quantum_moe")
    elif use_qufex:
        temp = QuFeX(temp, name="qufex")
    elif use_cmoe:
        # Đối chứng cổ điển của QuantumMoE (mục 5.6.2): cùng patch size đa tỷ
        # lệ + router + residual, nhưng expert là MLP cổ điển thay vì PQC.
        temp, _ = ClassicalMoE(temp, expert_layers=expert_layers, name="classical_moe")
    else:
        temp = layers.Conv2D(num_filters[-1] * 2, (3, 3), activation="relu",
                             name="bottleneck1", padding="same")(temp)
        temp = layers.Conv2D(num_filters[-1] * 2, (3, 3), activation="relu",
                             name="bottleneck2", padding="same")(temp)

    # --- LSS (tuỳ chọn) ---
    if use_lss:
        temp = LSS(temp, reduction=1, kernel_size=5, name="mamba_bottleneck")
        temp = layers.Activation("relu")(temp)

    # --- Decoder ---
    for f in num_filters[::-1]:
        temp = convT_block(temp, f, skip_conn.pop(), use_rdms=use_rdms)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(temp)

    model = tf.keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[iou_fn],
    )
    return model


# ---------------------------------------------------------------------------
# Bảng cấu hình các biến thể (tên biến thể -> công tắc)
# ---------------------------------------------------------------------------
VARIANTS = {
    "UNet":          dict(use_rdms=False, use_qmoe=False, use_lss=False),
    "UNet_LSS":      dict(use_rdms=False, use_qmoe=False, use_lss=True),
    "UNet_RDMS":     dict(use_rdms=True,  use_qmoe=False, use_lss=False),
    "UNet_RDMS_LSS": dict(use_rdms=True,  use_qmoe=False, use_lss=True),
    "QuMoE":         dict(use_rdms=False, use_qmoe=True,  use_lss=False),
    "QMoE_LSS":      dict(use_rdms=False, use_qmoe=True,  use_lss=True),
    "QMoE_RDMS":     dict(use_rdms=True,  use_qmoe=True,  use_lss=False),
    "HQMoSS_Net":    dict(use_rdms=True,  use_qmoe=True,  use_lss=True),
    # Baseline lượng tử kiểu QCNN (khác nhánh Quantum MoE): conv thường + QuFeX.
    "QuNet":         dict(use_rdms=False, use_qmoe=False, use_lss=False, use_qufex=True),
    # Đối chứng cổ điển của QuMoE (mục 5.6.2): cùng bộ khung + patch size đa tỷ
    # lệ + router + residual, nhưng expert là MLP cổ điển thay vì PQC lượng tử.
    "CMoE":          dict(use_rdms=False, use_qmoe=False, use_lss=False, use_cmoe=True),
}


def build_variant(variant, input_shape, num_filters, expert_layers=(5, 5, 5),
                  learning_rate=0.001, iou_fn=compute_iou):
    """Dựng mô hình theo tên biến thể trong bảng VARIANTS."""
    if variant not in VARIANTS:
        raise ValueError(f"Biến thể không hợp lệ: {variant}. Chọn một trong {list(VARIANTS)}.")
    return build_segmentation_model(
        input_shape, num_filters, expert_layers=expert_layers,
        learning_rate=learning_rate, iou_fn=iou_fn, name=variant, **VARIANTS[variant],
    )


# ---------------------------------------------------------------------------
# Các hàm dựng đặt tên rõ ràng (giữ đúng chữ ký như trong notebook gốc)
# ---------------------------------------------------------------------------
def HQMoSS_Net(input_shape, iou_fn, num_filters, expert_layers=(5, 5, 5)):
    """Mô hình đầy đủ: RDMS + Quantum MoE + LSS."""
    return build_variant("HQMoSS_Net", input_shape, num_filters,
                         expert_layers=expert_layers, iou_fn=iou_fn)


def QuMoE(input_shape, iou_fn, num_filters, expert_layers=(5, 5, 5)):
    """U-Net conv thường + Quantum MoE (dùng cho ablation số lớp mạch)."""
    return build_variant("QuMoE", input_shape, num_filters,
                         expert_layers=expert_layers, iou_fn=iou_fn)


def CMoE(input_shape, iou_fn, num_filters, expert_layers=(5, 5, 5)):
    """U-Net conv thường + Classical MoE -- đối chứng cổ điển của QuMoE."""
    return build_variant("CMoE", input_shape, num_filters,
                         expert_layers=expert_layers, iou_fn=iou_fn)


def UNet(input_shape, iou_fn, num_filters):
    """U-Net chuẩn tự custom."""
    return build_variant("UNet", input_shape, num_filters, iou_fn=iou_fn)


def UNet_RDMS_LSS(input_shape, iou_fn, num_filters):
    """U-Net chuẩn + RDMS + LSS (không có Quantum MoE)."""
    return build_variant("UNet_RDMS_LSS", input_shape, num_filters, iou_fn=iou_fn)


def QuNet(input_shape, iou_fn, num_filters):
    """U-Net conv thường + QuFeX (QCNN) ở bottleneck — baseline lượng tử so sánh."""
    return build_variant("QuNet", input_shape, num_filters, iou_fn=iou_fn)
