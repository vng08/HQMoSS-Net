"""
Mô-đun Classical Mixture-of-Experts (CMoE) -- đối chứng cổ điển của Quantum MoE.

Kiến trúc mirror chính xác QuantumMoE (cùng 3 patch size 2x2/2x4/8x8, cùng số
"lớp xử lý" mỗi expert, cùng router softmax + kết nối phần dư) nhưng thay mạch
lượng tử (PQC) bằng một khối MLP cổ điển (Dense stack). Dùng làm đối chứng cho
ablation số lớp mạch (mục 5.6.2): so QuantumMoE(5,5,5) với CMoE 5 lớp Dense để
xem lợi ích quan sát được có đến từ biểu diễn lượng tử hay chỉ từ cấu trúc
"chia patch đa tỷ lệ + router + residual".

  * ClassicalExpert : trích patch từ feature map, chuẩn hoá, chạy qua n_layers
                      lớp Dense (thay cho PQC), rồi ghép patch lại bằng
                      overlap-add -- logic giống hệt QuantumExpert.
  * ClassicalRouter : trộn đầu ra các expert bằng trọng số softmax phụ thuộc
                      đầu vào (giống hệt Router của QuantumMoE).
  * ClassicalMoE    : lắp 3 expert cổ điển (patch 2x2/2x4/8x8) + router + kết
                      nối phần dư.
"""

import tensorflow as tf
import tensorflow.keras.layers as layers


def ClassicalExpert(inputs, patch_size, hidden_dim, n_layers, name):
    """
    Một expert cổ điển xử lý feature map theo patch trượt (stride 1), đối
    chứng trực tiếp với QuantumExpert: cùng patch_size, cùng n_layers, nhưng
    "mạch" là n_layers lớp Dense thay vì PQC.
    """
    patch_h, patch_w = patch_size
    h, w, channels = int(inputs.shape[1]), int(inputs.shape[2]), int(inputs.shape[-1])
    patch_dim = patch_h * patch_w * channels

    pad_h, pad_w = max(patch_h - h, 0), max(patch_w - w, 0)
    ph, pw = h + pad_h, w + pad_w
    padded = tf.pad(inputs, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
    nh, nw = ph - patch_h + 1, pw - patch_w + 1

    patches = tf.image.extract_patches(
        padded,
        sizes=[1, patch_h, patch_w, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )

    batch = tf.shape(inputs)[0]
    patches = tf.reshape(patches, (-1, patch_dim))

    # LayerNorm -> n_layers lớp Dense (thay cho PQC) -> chiếu về patch_dim -> LayerNorm
    x = layers.LayerNormalization(axis=-1, name=f"{name}_pre_ln")(patches)
    for l in range(n_layers):
        x = layers.Dense(hidden_dim, activation="relu", name=f"{name}_dense{l}")(x)
    x = layers.Dense(patch_dim, activation=None, name=f"{name}_proj_out")(x)
    x = layers.LayerNormalization(axis=-1, name=f"{name}_post_ln")(x)

    x = tf.reshape(x, (batch, nh, nw, patch_h, patch_w, channels))

    # Overlap-add: cộng chồng các patch rồi chia số lần phủ (giống QuantumExpert)
    out = tf.zeros_like(padded)
    count = tf.zeros_like(padded)
    for i in range(patch_h):
        for j in range(patch_w):
            part = x[:, :, :, i, j, :]
            paddings = [[0, 0], [i, ph - nh - i], [j, pw - nw - j], [0, 0]]
            out = out + tf.pad(part, paddings)
            count = count + tf.pad(tf.ones_like(part), paddings)

    feature = out / tf.maximum(count, tf.cast(1e-8, count.dtype))
    return feature[:, :h, :w, :]


def ClassicalRouter(inputs, experts, name="classical_router"):
    """Trộn đầu ra các expert cổ điển bằng trọng số softmax -- giống hệt Router lượng tử."""
    x = layers.Concatenate(name=f"{name}_concat")([inputs] + experts)
    logits = layers.Conv2D(len(experts), 1, name=f"{name}_logits")(x)
    weights = layers.Softmax(axis=-1, name=f"{name}_softmax")(logits)
    stacked = tf.stack(experts, axis=-1)
    mixed = tf.reduce_sum(stacked * tf.expand_dims(weights, axis=-2), axis=-1)
    return mixed, weights


def ClassicalMoE(inputs, expert_layers=(5, 5, 5), name="classical_moe"):
    """
    Classical Mixture-of-Experts: đối chứng cổ điển của QuantumMoE, cùng 3
    patch size đa tỉ lệ (2x2/2x4/8x8) + router + kết nối phần dư.

    hidden_dim của mỗi expert (32/64/512) tăng theo patch_dim của mỗi nhánh
    (không cố ép bằng số tham số lượng tử) -- mục tiêu là một MLP đủ dùng ở
    quy mô cổ điển thông thường, không phải một baseline bị đói tham số.
    """
    e1 = ClassicalExpert(inputs, (2, 2), hidden_dim=32, n_layers=expert_layers[0], name=f"{name}_expert1")
    e2 = ClassicalExpert(inputs, (2, 4), hidden_dim=64, n_layers=expert_layers[1], name=f"{name}_expert2")
    e3 = ClassicalExpert(inputs, (8, 8), hidden_dim=512, n_layers=expert_layers[2], name=f"{name}_expert3")

    mixed_output, router_weights = ClassicalRouter(inputs, [e1, e2, e3], name=f"{name}_router")
    output = layers.Add(name=f"{name}_residual")([inputs, mixed_output])
    return output, router_weights
