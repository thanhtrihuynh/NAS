import tensorflow as tf


WEIGHT_LAYER_TYPES = (
    tf.keras.layers.Conv1D,
    tf.keras.layers.SeparableConv1D,
    tf.keras.layers.Dense,
)


def quant_limits(bits):
    bits = int(bits)

    if bits == 8:
        return -127.0, 127.0

    if bits == 4:
        return -7.0, 7.0

    raise ValueError(
        f"Chỉ hỗ trợ INT4 hoặc INT8, nhận được: {bits}"
    )


def fake_quant_ste(x, bits):
    """
    Symmetric fake quantization + Straight-Through Estimator.

    Forward:
        FP32 -> quantize -> dequantize

    Backward:
        gradient đi xuyên qua phép lượng tử hóa.
    """

    qmin, qmax = quant_limits(bits)

    x = tf.convert_to_tensor(x)

    max_abs = tf.reduce_max(
        tf.abs(x)
    )

    scale = tf.maximum(
        max_abs / qmax,
        tf.cast(1e-8, x.dtype)
    )

    q = tf.round(
        x / scale
    )

    q = tf.clip_by_value(
        q,
        tf.cast(qmin, x.dtype),
        tf.cast(qmax, x.dtype)
    )

    x_quant = q * scale

    return (
        x
        + tf.stop_gradient(
            x_quant - x
        )
    )


@tf.keras.utils.register_keras_serializable(
    package="ECGQAT"
)
class QATWeightWrapper(
    tf.keras.layers.Wrapper
):
    def __init__(
        self,
        layer,
        bits=8,
        **kwargs
    ):
        super().__init__(
            layer,
            **kwargs
        )

        self.bits = int(bits)

        if self.bits not in (4, 8):
            raise ValueError(
                "bits phải là 4 hoặc 8"
            )

    def _dense_call(
        self,
        inputs
    ):
        kernel_q = fake_quant_ste(
            self.layer.kernel,
            self.bits
        )

        outputs = tf.linalg.matmul(
            inputs,
            kernel_q
        )

        if self.layer.use_bias:
            outputs = (
                outputs
                + self.layer.bias
            )

        if self.layer.activation is not None:
            outputs = (
                self.layer.activation(
                    outputs
                )
            )

        return outputs

    def _conv1d_call(
        self,
        inputs
    ):
        if self.layer.data_format not in (
            None,
            "channels_last"
        ):
            raise NotImplementedError(
                "Project hiện tại chỉ hỗ trợ "
                "Conv1D channels_last."
            )

        kernel_q = fake_quant_ste(
            self.layer.kernel,
            self.bits
        )

        # convolution_op giữ nguyên:
        # stride, dilation, padding, groups...
        outputs = (
            self.layer.convolution_op(
                inputs,
                kernel_q
            )
        )

        if self.layer.use_bias:
            outputs = (
                outputs
                + self.layer.bias
            )

        if self.layer.activation is not None:
            outputs = (
                self.layer.activation(
                    outputs
                )
            )

        return outputs

    def _separable_conv1d_call(
        self,
        inputs
    ):
        if self.layer.data_format not in (
            None,
            "channels_last"
        ):
            raise NotImplementedError(
                "Project hiện tại chỉ hỗ trợ "
                "SeparableConv1D channels_last."
            )

        depthwise_q = fake_quant_ste(
            self.layer.depthwise_kernel,
            self.bits
        )

        pointwise_q = fake_quant_ste(
            self.layer.pointwise_kernel,
            self.bits
        )

        outputs = tf.keras.ops.separable_conv(
            inputs,
            depthwise_q,
            pointwise_q,
            strides=self.layer.strides,
            padding=self.layer.padding,
            data_format=self.layer.data_format,
            dilation_rate=self.layer.dilation_rate
        )

        if self.layer.use_bias:
            outputs = (
                outputs
                + self.layer.bias
            )

        if self.layer.activation is not None:
            outputs = (
                self.layer.activation(
                    outputs
                )
            )

        return outputs

    def call(
        self,
        inputs,
        training=None,
        mask=None,
        **kwargs
    ):
        if isinstance(
            self.layer,
            tf.keras.layers.Dense
        ):
            return self._dense_call(
                inputs
            )

        if isinstance(
            self.layer,
            tf.keras.layers.Conv1D
        ):
            return self._conv1d_call(
                inputs
            )

        if isinstance(
            self.layer,
            tf.keras.layers.SeparableConv1D
        ):
            return (
                self._separable_conv1d_call(
                    inputs
                )
            )

        raise TypeError(
            "QATWeightWrapper không hỗ trợ "
            f"layer {type(self.layer)}"
        )

    def get_config(self):
        config = super().get_config()

        config.update({
            "bits": self.bits
        })

        return config


@tf.keras.utils.register_keras_serializable(
    package="ECGQAT"
)
class QATActivationWrapper(
    tf.keras.layers.Wrapper
):
    def __init__(
        self,
        layer,
        bits=8,
        **kwargs
    ):
        super().__init__(
            layer,
            **kwargs
        )

        self.bits = int(bits)

        if self.bits not in (4, 8):
            raise ValueError(
                "bits phải là 4 hoặc 8"
            )

    def call(
        self,
        inputs,
        training=None,
        mask=None,
        **kwargs
    ):
        try:
            outputs = self.layer(
                inputs,
                training=training,
                **kwargs
            )

        except TypeError:
            outputs = self.layer(
                inputs,
                **kwargs
            )

        return fake_quant_ste(
            outputs,
            self.bits
        )

    def get_config(self):
        config = super().get_config()

        config.update({
            "bits": self.bits
        })

        return config


def get_activation_controller(
    layer_name
):
    if layer_name == "stem_act":
        return "stem_conv"

    if layer_name == "head_act":
        return "head_dense"

    mappings = [
        (
            "_b1_act",
            "_b1_sepconv"
        ),
        (
            "_b2_act",
            "_b2_sepconv"
        ),
        (
            "_b3_act",
            "_b3_sepconv"
        ),
        (
            "_pool_act",
            "_pool_proj"
        ),
        (
            "_out_bn",
            "_out_proj"
        ),
        (
            "_shortcut_bn",
            "_shortcut"
        ),
    ]

    for suffix, target_suffix in mappings:
        if layer_name.endswith(
            suffix
        ):
            prefix = layer_name[
                :-len(suffix)
            ]

            return (
                prefix
                + target_suffix
            )

    return None


def get_precision(
    layer_name,
    precision_config,
    default_bits=8
):
    bits = int(
        precision_config.get(
            layer_name,
            default_bits
        )
    )

    if bits not in (4, 8):
        raise ValueError(
            f"{layer_name}: "
            "precision phải là 4 hoặc 8"
        )

    return bits


def build_qat_model(
    base_model,
    precision_config
):
    """
    Tạo model QAT từ model FP32 đã train.

    Conv/SepConv/Dense:
        fake quant weight trong forward.

    Activation:
        fake quant theo precision
        của computational layer tương ứng.
    """

    def clone_function(
        layer
    ):
        cloned_layer = (
            layer.__class__.from_config(
                layer.get_config()
            )
        )

        if isinstance(
            layer,
            WEIGHT_LAYER_TYPES
        ):
            bits = get_precision(
                layer.name,
                precision_config
            )

            return QATWeightWrapper(
                cloned_layer,
                bits=bits,
                name=f"qat_w_{layer.name}"
            )

        controller = (
            get_activation_controller(
                layer.name
            )
        )

        if controller is not None:
            bits = get_precision(
                controller,
                precision_config
            )

            return QATActivationWrapper(
                cloned_layer,
                bits=bits,
                name=f"qat_a_{layer.name}"
            )

        return cloned_layer

    qat_model = (
        tf.keras.models.clone_model(
            base_model,
            clone_function=clone_function
        )
    )

    target_layers = {}

    for layer in qat_model.layers:
        if isinstance(
            layer,
            (
                QATWeightWrapper,
                QATActivationWrapper
            )
        ):
            target_layers[
                layer.layer.name
            ] = layer.layer

        else:
            target_layers[
                layer.name
            ] = layer

    # Copy toàn bộ FP32 weights sang model QAT.
    for original_layer in (
        base_model.layers
    ):
        if (
            original_layer.name
            not in target_layers
        ):
            continue

        original_weights = (
            original_layer.get_weights()
        )

        if len(
            original_weights
        ) == 0:
            continue

        target_layer = (
            target_layers[
                original_layer.name
            ]
        )

        target_layer.set_weights(
            original_weights
        )

    return qat_model