import keras_tuner as kt
import tensorflow as tf

def check_model_compute_budget(model: tf.keras.Model,
                               max_maccs: int,
                               max_num_params: int):
    maccs = get_model_maccs(model)
    num_params = get_model_num_params(model)

    print("[INFO] MACCs: {}, # Params: {}".format(maccs, num_params))

    return maccs <= max_maccs and num_params <= max_num_params

def get_model_num_params(model: tf.keras.Model):
    return model.count_params()

def get_layer_maccs(layer, input_shape=None):
    """Estimate MACCs for common layer types."""
    if isinstance(layer, tf.keras.layers.Conv2D):
        # MACCs = kernel_height * kernel_width * in_channels * out_channels * output_height * output_width
        in_channels = layer.input.shape[-1]
        out_channels = layer.filters
        kh, kw = layer.kernel_size
        oh, ow = layer.output.shape[1:3]
        return kh * kw * in_channels * out_channels * oh * ow
    elif isinstance(layer, tf.keras.layers.Dense):
        # MACCs = input_units * output_units
        input_units = layer.input.shape[-1]
        output_units = layer.units
        return input_units * output_units
    elif isinstance(layer, (tf.keras.layers.DepthwiseConv2D)):
        in_channels = layer.input.shape[-1]
        kh, kw = layer.kernel_size
        oh, ow = layer.output.shape[1:3]
        # depthwise: one filter per input channel
        return kh * kw * in_channels * oh * ow
    else:
        # for pooling, activation, batchnorm, etc., MACCs are negligible
        return 0

def get_model_maccs(model: tf.keras.Model) -> int:
    """Returns approximate MACCs (multiply-accumulate operations) for the model."""
    total_maccs = 0
    for layer in model.layers:
        total_maccs += get_layer_maccs(layer)
    return total_maccs
