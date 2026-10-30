"""PruDensNet, written out from the executed notebook
`notebooks/test-acc-96-05-14-and-validation-acc-97-27.ipynb`, the run that
produced the published 96.05% test / 97.27% validation numbers.

The real model is 1,130,382 trainable parameters. The paper's 1,461,587 figure
includes a ParamPad layer of 331,205 inert weights, multiplied by 0.0 in the
forward pass, zero gradient. The paper discloses it as a no-op layer. The pad
is off by default here; pass pad_to=1_461_587 to reproduce the padded count.

Architecture: stem Conv 48 -> Conv 64 -> MaxPool, then three depthwise-separable
blocks at widths 128 / 320 / 640, each block being two SeparableConv layers with
GroupNorm, GELU, Squeeze-Excite and a spatial gate, each followed by MaxPool,
then GAP -> Dense 256 (GELU, Dropout 0.25) -> softmax over 4 classes.
Input 256x256x3, GaussianNoise(0.008) on the input during training.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import gelu
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    SeparableConv2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, Activation, GaussianNoise, Input,
    Reshape, Multiply, Layer, Concatenate,
)

REAL_TRAINABLE_PARAMS = 1_130_382
PUBLISHED_PADDED_PARAMS = 1_461_587


class GroupNormalization(Layer):
    """GroupNorm as implemented in the notebook (TFA was unavailable).

    Groups are reduced until they divide the channel count."""

    def __init__(self, groups=8, axis=-1, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups = int(groups)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = int(input_shape[self.axis])
        g = min(self.groups, dim)
        while dim % g != 0 and g > 1:
            g -= 1
        self.groups = g
        self.gamma = self.add_weight(shape=(dim,), initializer="ones",
                                     trainable=True, name="gamma")
        self.beta = self.add_weight(shape=(dim,), initializer="zeros",
                                    trainable=True, name="beta")
        super().build(input_shape)

    def call(self, x):
        N, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        G = self.groups
        x = tf.reshape(x, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, axes=[1, 2, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.epsilon)
        x = tf.reshape(x, [N, H, W, C])
        return x * self.gamma + self.beta


class SqueezeExcite(Layer):
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        c = int(input_shape[-1])
        hidden = max(8, c // self.ratio)
        self.fc1 = Dense(hidden, activation="relu")
        self.fc2 = Dense(c, activation="sigmoid")
        super().build(input_shape)

    def call(self, x):
        s = GlobalAveragePooling2D()(x)
        s = self.fc2(self.fc1(s))
        s = Reshape((1, 1, -1))(s)
        return Multiply()([x, s])


class SpatialGate(Layer):
    """CBAM-style spatial attention: 7x7 conv over [avg, max] channel maps."""

    def build(self, input_shape):
        self.conv = Conv2D(1, kernel_size=7, padding="same", use_bias=True)
        super().build(input_shape)

    def call(self, x):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx = tf.reduce_max(x, axis=-1, keepdims=True)
        s = self.conv(Concatenate(axis=-1)([avg, mx]))
        return Multiply()([x, tf.nn.sigmoid(s)])


class ParamPad(Layer):
    """n_params trainable weights whose output contribution is multiplied by
    0.0. Zero gradient, zero effect on the function; raises the reported
    trainable-parameter count and nothing else."""

    def __init__(self, n_params=0, **kwargs):
        super().__init__(**kwargs)
        self.n_params = int(n_params)

    def build(self, input_shape):
        if self.n_params > 0:
            self.pad_w = self.add_weight(
                name="param_pad", shape=(self.n_params,),
                initializer="zeros", trainable=True, dtype=self.dtype)
        super().build(input_shape)

    def call(self, x):
        if hasattr(self, "pad_w"):
            return x + tf.reduce_sum(self.pad_w) * tf.cast(0.0, x.dtype)
        return x


def trainable_param_count(model):
    return int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))


def _gn():
    return GroupNormalization(groups=8, axis=-1, epsilon=1e-5)


def _conv_gn_act(x, filters, k=3, s=1):
    x = Conv2D(filters, (k, k), strides=s, padding="same", use_bias=False)(x)
    return Activation(gelu)(_gn()(x))


def _sep_block(x, filters, se_ratio=16):
    for _ in range(2):
        x = SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
        x = Activation(gelu)(_gn()(x))
        x = SqueezeExcite(se_ratio)(x)
        x = SpatialGate()(x)
    return x


def build_prudensnet(num_classes=4, img_size=(256, 256), pad_to=None):
    """Build the real model. pad_to=PUBLISHED_PADDED_PARAMS reproduces the
    padded count the paper reports."""
    inp = Input(shape=(img_size[0], img_size[1], 3), name="input")
    x = GaussianNoise(0.008)(inp)
    x = _conv_gn_act(x, 48)
    x = _conv_gn_act(x, 64)
    x = MaxPooling2D(2)(x)
    for filters in (128, 320, 640):
        x = _sep_block(x, filters)
        x = MaxPooling2D(2)(x)
    x = GlobalAveragePooling2D(name="GAP")(x)
    x = Dense(256, use_bias=True)(x)
    x = Activation(gelu)(x)
    x = Dropout(0.25)(x)
    out = Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = Model(inp, out, name="prudensnet")

    real = trainable_param_count(model)
    if num_classes == 4 and img_size == (256, 256):
        assert real == REAL_TRAINABLE_PARAMS, (
            f"expected {REAL_TRAINABLE_PARAMS:,} real trainables, got {real:,}")

    if pad_to is not None:
        if real > pad_to:
            raise ValueError(f"base ({real:,}) exceeds pad target ({pad_to:,})")
        if pad_to > real:
            out = ParamPad(n_params=pad_to - real, name="param_pad")(out)
            model = Model(inp, out, name=model.name)
    return model


if __name__ == "__main__":
    m = build_prudensnet()
    print(f"real trainables:   {trainable_param_count(m):,}")
    m = build_prudensnet(pad_to=PUBLISHED_PADDED_PARAMS)
    print(f"padded trainables: {trainable_param_count(m):,} (the paper's figure)")
