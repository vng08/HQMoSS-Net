"""
Mô-đun Quantum Mixture-of-Experts (QMoE).

Ba expert lượng tử, mỗi expert là một mạch tham số hoá (PQC) chạy trên các patch
có kích thước khác nhau (đa tỉ lệ), được trộn lại bằng một router học được:

  * PQCExpert   : định nghĩa mạch lượng tử tham số hoá (Hadamard -> các lớp Rot +
                  CNOT vòng -> lớp xoay RX/RY cuối).
  * QuantumExpert: trích patch từ feature map, nhúng biên độ (amplitude embedding),
                  chạy qua PQC, rồi ghép patch trở lại bằng overlap-add.
  * Router      : trộn đầu ra các expert bằng trọng số softmax phụ thuộc đầu vào.
  * QuantumMoE  : lắp 3 expert (patch 2x2/2x4/8x8) + router + kết nối phần dư.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import pennylane as qml


def PQCExpert(n_qubits, weight_0, weight_1):
    """Mạch lượng tử tham số hoá dùng chung cho mọi expert."""
    for i in range(n_qubits):
        qml.Hadamard(wires=i)

    n_layers = weight_0.shape[0]
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.Rot(weight_0[layer][i][0], weight_0[layer][i][1], weight_0[layer][i][2], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])

    for i in range(n_qubits):
        if i % 3 == 0:
            qml.RX(weight_1[i], wires=i)
        elif i % 3 == 1:
            qml.RY(weight_1[i], wires=i)


def QuantumExpert(inputs, patch_size, n_qubits, n_layers, pqc_fn, name):
    """
    Một expert lượng tử xử lý feature map theo patch trượt (stride 1).

    Quy trình: pad -> extract_patches -> LayerNorm -> mạch lượng tử (amplitude
    embedding + PQC + probs) -> LayerNorm -> ghép patch lại bằng overlap-add ->
    cắt bỏ vùng pad để trả về đúng kích thước (H, W, C) ban đầu.
    """
    patch_h, patch_w = patch_size
    h, w, channels = int(inputs.shape[1]), int(inputs.shape[2]), int(inputs.shape[-1])
    patch_dim = patch_h * patch_w * channels
    quantum_dim = 2 ** n_qubits

    pad_h, pad_w = max(patch_h - h, 0), max(patch_w - w, 0)
    ph, pw = h + pad_h, w + pad_w
    padded = tf.pad(inputs, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
    nh, nw = ph - patch_h + 1, pw - patch_w + 1

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="tf")
    def circuit(inputs, weight_0, weight_1):
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
        pqc_fn(n_qubits, weight_0, weight_1)
        return qml.probs(wires=range(n_qubits))

    weight_shapes = {"weight_0": (n_layers, n_qubits, 3), "weight_1": n_qubits}
    qlayer = qml.qnn.KerasLayer(circuit, weight_shapes, output_dim=quantum_dim, name=f"{name}_circuit")

    patches = tf.image.extract_patches(
        padded,
        sizes=[1, patch_h, patch_w, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )

    batch = tf.shape(inputs)[0]
    patches = tf.reshape(patches, (-1, patch_dim))

    # LayerNorm -> PQC -> probs -> LayerNorm
    patches = layers.LayerNormalization(axis=-1, name=f"{name}_pre_ln")(patches)
    patches = qlayer(patches)
    patches = layers.LayerNormalization(axis=-1, name=f"{name}_post_ln")(patches)

    # Bỏ phần vector dư và đưa về dạng patch
    patches = patches[:, :patch_dim]
    patches = tf.reshape(patches, (batch, nh, nw, patch_h, patch_w, channels))

    # Overlap-add: cộng chồng các patch rồi chia số lần phủ
    out = tf.zeros_like(padded)
    count = tf.zeros_like(padded)
    for i in range(patch_h):
        for j in range(patch_w):
            part = patches[:, :, :, i, j, :]
            paddings = [[0, 0], [i, ph - nh - i], [j, pw - nw - j], [0, 0]]
            out = out + tf.pad(part, paddings)
            count = count + tf.pad(tf.ones_like(part), paddings)

    feature = out / tf.maximum(count, tf.cast(1e-8, count.dtype))
    return feature[:, :h, :w, :]  # cắt bỏ vùng pad


def Router(inputs, experts, name="router"):
    """Trộn đầu ra các expert bằng trọng số softmax theo từng pixel."""
    x = layers.Concatenate(name=f"{name}_concat")([inputs] + experts)
    logits = layers.Conv2D(len(experts), 1, name=f"{name}_logits")(x)
    weights = layers.Softmax(axis=-1, name=f"{name}_softmax")(logits)
    stacked = tf.stack(experts, axis=-1)
    mixed = tf.reduce_sum(stacked * tf.expand_dims(weights, axis=-2), axis=-1)
    return mixed, weights


def QuantumMoE(inputs, expert_layers=(5, 5, 5), name="quantum_moe"):
    """
    Quantum Mixture-of-Experts: 3 expert đa tỉ lệ + router + kết nối phần dư.

    expert_layers: số lớp mạch của từng expert (dùng cho ablation số lớp mạch,
    ví dụ (1,1,1) ... (5,5,5)).
    """
    e1 = QuantumExpert(inputs, (2, 2), 5, expert_layers[0], PQCExpert, f"{name}_expert1")
    e2 = QuantumExpert(inputs, (2, 4), 6, expert_layers[1], PQCExpert, f"{name}_expert2")
    e3 = QuantumExpert(inputs, (8, 8), 9, expert_layers[2], PQCExpert, f"{name}_expert3")

    mixed_output, router_weights = Router(inputs, [e1, e2, e3], name=f"{name}_router")
    output = layers.Add(name=f"{name}_residual")([inputs, mixed_output])
    return output, router_weights


# ---------------------------------------------------------------------------
# QuFeX: Quantum Feature eXtraction (mô hình lượng tử so sánh, kiểu QCNN)
# ---------------------------------------------------------------------------
# Đây là một *baseline lượng tử* khác với Quantum MoE: thay vì trộn nhiều expert
# đa tỉ lệ bằng router, QuFeX chia feature map ở bottleneck thành nhiều nhóm kênh
# (splits), trích patch KHÔNG chồng lấn rồi cho qua một mạch QCNN tham số hoá
# (PQC1) đo kỳ vọng PauliZ, cuối cùng ghép lại và cộng phần dư. Dùng trong nghiên
# cứu so sánh (mô hình "QuNet" / thư mục kết quả "Qunet_12_1").
def PQC1(n_qubits, weight_0, weight_1):
    """Mạch QCNN kiểu 1: 2 tầng entangling (RX/RZ + CNOT, rồi RX/RY + CNOT) xen kẽ CZ."""
    indices = int(n_qubits / 2)

    # U1
    for i in range(indices):
        qml.RX(weight_0[0], wires=2 * i)
        qml.RZ(weight_0[1], wires=2 * i + 1)
        qml.CNOT(wires=[2 * i, 2 * i + 1])
    for i in range(1, indices):
        qml.RX(weight_0[0], wires=2 * i - 1)
        qml.RZ(weight_0[1], wires=2 * i)
        qml.CNOT(wires=[2 * i - 1, 2 * i])

    # V1
    for i in range(indices):
        qml.CZ(wires=[2 * i, 2 * i + 1])

    indices = int(indices / 2)

    # U2
    for i in range(indices):
        qml.RX(weight_1[0], wires=2 * (2 * i))
        qml.RY(weight_1[1], wires=2 * (2 * i + 1))
        qml.CNOT(wires=[2 * (2 * i), 2 * (2 * i + 1)])
    for i in range(1, indices):
        qml.RX(weight_1[0], wires=2 * (2 * i - 1))
        qml.RY(weight_1[1], wires=2 * (2 * i))
        qml.CNOT(wires=[2 * (2 * i - 1), 2 * (2 * i)])

    # V2
    for i in range(indices):
        qml.CZ(wires=[2 * (2 * i), 2 * (2 * i + 1)])


def PQC2(n_qubits, weight_0, weight_1):
    """Mạch QCNN kiểu 2: hoán vị cổng xoay (RY/RX rồi RZ/RX) so với PQC1."""
    indices = int(n_qubits / 2)

    # U1
    for i in range(indices):
        qml.RY(weight_0[0], wires=2 * i)
        qml.RX(weight_0[1], wires=2 * i + 1)
        qml.CNOT(wires=[2 * i, 2 * i + 1])
    for i in range(1, indices):
        qml.RY(weight_0[0], wires=2 * i - 1)
        qml.RX(weight_0[1], wires=2 * i)
        qml.CNOT(wires=[2 * i - 1, 2 * i])

    # V1
    for i in range(indices):
        qml.CZ(wires=[2 * i, 2 * i + 1])

    indices = int(indices / 2)

    # U2
    for i in range(indices):
        qml.RZ(weight_1[0], wires=2 * (2 * i))
        qml.RX(weight_1[1], wires=2 * (2 * i + 1))
        qml.CNOT(wires=[2 * (2 * i), 2 * (2 * i + 1)])
    for i in range(1, indices):
        qml.RZ(weight_1[0], wires=2 * (2 * i - 1))
        qml.RX(weight_1[1], wires=2 * (2 * i))
        qml.CNOT(wires=[2 * (2 * i - 1), 2 * (2 * i)])

    # V2
    for i in range(indices):
        qml.CZ(wires=[2 * (2 * i), 2 * (2 * i + 1)])


def QuFeX(inputs, splits=None, n_qubits=12, patch_size=(3, 4), q_filters=1,
          pqc_fn=PQC1, name="qufex"):
    """
    Khối QuFeX ở bottleneck: chia kênh -> trích patch không chồng lấn -> QCNN ->
    ghép lại -> cộng phần dư.

    splits    : số nhóm kênh chia ra. Mặc định = (số kênh đầu vào // q_filters),
                đúng như cấu hình "12_1" trong notebook (c_last=8, q_filters=1 -> 8).
    n_qubits  : số qubit của mạch QCNN (12 cho biến thể 12_1).
    patch_size: (patch_h, patch_w) cho patch không chồng lấn (3x4 -> 12 = n_qubits).
    """
    channels_in = int(inputs.shape[-1])
    if splits is None:
        splits = max(channels_in // q_filters, 1)

    patch_height, patch_width = patch_size
    channels = max(n_qubits // splits, 1)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="tf")
    def qcnn1(inputs, weight_0, weight_1):
        qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits), rotation="Y")
        pqc_fn(n_qubits, weight_0, weight_1)
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    weight_shapes = {"weight_0": 2, "weight_1": 2}
    qcnn_layer1 = qml.qnn.KerasLayer(qcnn1, weight_shapes, output_dim=n_qubits, name=f"{name}_qcnn1")

    q_split = tf.split(inputs, num_or_size_splits=splits, axis=-1)

    q = []
    for i in range(splits):
        patches = tf.image.extract_patches(
            images=q_split[i],
            sizes=[1, patch_height, patch_width, 1],
            strides=[1, patch_height, patch_width, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        _, new_height, new_width, flattened_patch_size = patches.shape
        patches = layers.Reshape((new_height * new_width, flattened_patch_size))(patches)
        patches = tf.reshape(patches, (-1, flattened_patch_size))

        q0 = qcnn_layer1(patches)
        q0 = tf.reshape(q0, (-1, new_height, new_width, patch_height, patch_width, channels))
        q0 = tf.transpose(q0, perm=[0, 1, 3, 2, 4, 5])
        restored_fm = tf.reshape(q0, (-1, new_height * patch_height, new_width * patch_width, channels))
        q.append(restored_fm)

    q_res = layers.Concatenate(axis=-1, name=f"{name}_resultant")(q)
    q_out = layers.Add(name=f"{name}_residual")([inputs, q_res])
    return q_out
