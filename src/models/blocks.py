"""
Các khối kiến trúc cổ điển (không lượng tử) dùng để lắp ráp mô hình phân vùng:

  * RDMS (Residual Dilated Multi-Scale): khối tích chập đa tỉ lệ với các nhánh
    depthwise dilated (dilation 1/2/3) + kết nối tắt phần dư.
  * conv_block / convT_block: khối encoder / decoder ở 2 phiên bản
      - plain: 2 lớp Conv2D thường (U-Net chuẩn),
      - rdms : dùng khối RDMS (HQMoSS-Net và các biến thể +RDMS).
  * SelectiveScan + LSS (Lightweight Selective Scan): khối quét chọn lọc kiểu
    Mamba, nắm bắt phụ thuộc không gian tầm xa ở tầng bottleneck.
"""

import tensorflow as tf
import tensorflow.keras.layers as layers


# ---------------------------------------------------------------------------
# 1. RDMS: Residual Dilated Multi-Scale block
# ---------------------------------------------------------------------------
def RDMS(x, num_filters):
    """Khối tích chập đa tỉ lệ có kết nối phần dư."""
    shortcut = layers.Conv2D(num_filters, (3, 3), padding="same")(x)

    b1 = layers.Activation("relu")(
        layers.DepthwiseConv2D((3, 3), dilation_rate=1, padding="same")(shortcut))
    b2 = layers.Activation("relu")(
        layers.DepthwiseConv2D((3, 3), dilation_rate=2, padding="same")(shortcut))
    b3 = layers.Activation("relu")(
        layers.DepthwiseConv2D((3, 3), dilation_rate=3, padding="same")(shortcut))

    x = layers.Concatenate(axis=-1)([b1, b2, b3])
    x = layers.Conv2D(num_filters, (3, 3), activation="relu", padding="same")(x)

    x = layers.Add()([shortcut, x])
    return layers.Activation("relu")(x)


# ---------------------------------------------------------------------------
# 2. Khối encoder / decoder (2 phiên bản: plain và rdms)
# ---------------------------------------------------------------------------
def conv_block(x, num_filters, use_rdms=False):
    """Khối encoder: trả về (feature giữ lại cho skip, feature đã downsample)."""
    if use_rdms:
        c = RDMS(x, num_filters)
    else:
        c = layers.Conv2D(num_filters, (3, 3), activation="relu", padding="same")(x)
        c = layers.Conv2D(num_filters, (3, 3), activation="relu", padding="same")(c)
    p = layers.MaxPooling2D((2, 2))(c)
    return c, p


def convT_block(x, num_filters, skip, use_rdms=False):
    """Khối decoder: upsample bằng Conv2DTranspose rồi ghép skip connection."""
    x = layers.Conv2DTranspose(num_filters, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.Concatenate(axis=-1)([x, skip])
    if use_rdms:
        x = RDMS(x, num_filters)
    else:
        x = layers.Conv2D(num_filters, (3, 3), activation="relu", padding="same")(x)
        x = layers.Conv2D(num_filters, (3, 3), activation="relu", padding="same")(x)
    return x


# ---------------------------------------------------------------------------
# 3. LSS: Lightweight Selective Scan block (lấy cảm hứng từ Mamba)
# ---------------------------------------------------------------------------
class SelectiveScan(layers.Layer):
    """Quét chọn lọc tuyến tính theo chuỗi. Input/Output: (B, N, C)."""

    def build(self, input_shape):
        c = int(input_shape[-1])
        self.log_a = self.add_weight(
            name="log_a", shape=(c,),
            initializer=tf.keras.initializers.Constant(-1.0), trainable=True,
        )
        self.b_proj = layers.Dense(c, activation="sigmoid", name="b_proj")
        self.c_proj = layers.Dense(c, activation="sigmoid", name="c_proj")
        super().build(input_shape)

    def call(self, x):
        a = tf.exp(-tf.nn.softplus(self.log_a))
        u = self.b_proj(x) * x
        gate = self.c_proj(x)

        u = tf.transpose(u, [1, 0, 2])
        h0 = tf.zeros_like(u[0])

        def step(h, u_t):
            return h * a + u_t

        h = tf.scan(step, u, initializer=h0)
        h = tf.transpose(h, [1, 0, 2])
        return h * gate


def LSS(x, reduction=2, kernel_size=5, name="lss"):
    """
    Khối Lightweight Selective Scan cho bottleneck 2D.
    Chiếu vào -> tách nhánh feature/gate -> quét chọn lọc theo chuỗi phẳng ->
    cổng sigmoid -> chiếu ra, cộng kết nối tắt.
    """
    c = int(x.shape[-1])
    hidden = max(c // reduction, 4)
    shortcut = x

    y = layers.LayerNormalization(axis=-1, name=f"{name}_ln")(x)
    y = layers.Conv2D(hidden * 2, 1, padding="same", use_bias=False, name=f"{name}_in_proj")(y)

    y_feat = layers.Lambda(lambda t: t[..., :hidden])(y)
    y_gate = layers.Lambda(lambda t: t[..., hidden:])(y)

    y_feat = layers.Conv2D(hidden, kernel_size, padding="same", use_bias=False)(y_feat)
    y_feat = layers.Activation("swish")(y_feat)

    y_seq = layers.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], tf.shape(t)[1] * tf.shape(t)[2], tf.shape(t)[3]])
    )(y_feat)
    y_seq = SelectiveScan(name=f"{name}_scan")(y_seq)

    y_feat = layers.Lambda(
        lambda t: tf.reshape(
            t[0], [tf.shape(t[1])[0], tf.shape(t[1])[1], tf.shape(t[1])[2], tf.shape(t[0])[-1]]
        )
    )([y_seq, x])

    y_gate = layers.Activation("sigmoid")(y_gate)
    y = layers.Multiply(name=f"{name}_selective_gate")([y_feat, y_gate])

    y = layers.Conv2D(
        c, 1, padding="same", use_bias=False,
        kernel_initializer="zeros", name=f"{name}_out_proj",
    )(y)

    return layers.Add()([shortcut, y])
