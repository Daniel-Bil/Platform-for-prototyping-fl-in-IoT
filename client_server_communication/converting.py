import numpy as np


def simple_quantize_floats(weights_list: list):
    quantized_weights_list = []
    for weight in weights_list:
        quantized_weights = weight.astype(np.float16)
        quantized_weights_list.append(quantized_weights)
    return quantized_weights_list


# Simple dequantization (float16)
def simple_dequantize_floats(quantized_weights_list: list):
    dequantized_weights_list = []
    for quantized_weights in quantized_weights_list:
        dequantized_weights = quantized_weights.astype(np.float32)
        dequantized_weights_list.append(dequantized_weights)
    return dequantized_weights_list


# Quantize weights to int8 with normalization
def quantize_weights_int(weights: list) -> tuple[list[np.ndarray], list[dict]]:
    quantized_weights = []
    params = []
    for weight in weights:
        mean = np.mean(weight)
        std_dev = np.std(weight)

        # Define clipping thresholds
        clip_min = mean - 2 * std_dev
        clip_max = mean + 2 * std_dev

        # Clip data
        clipped_data = np.clip(weight, clip_min, clip_max)
        max1 = np.max(clipped_data)
        min1 = np.min(clipped_data)

        # Normalize and quantize
        norm_data = 2 * ((clipped_data - min1) / (max1 - min1)) - 1
        quant_data = np.round(127 * norm_data).astype(np.int8)

        param = {'min': float(min1), 'max': float(max1)}
        quantized_weights.append(quant_data)
        params.append(param)

    return quantized_weights, params


# Dequantize int8 weights back to float32
def dequantize_weights_int(quantized_weights: list, params: list[dict]) -> list:
    dequantized_weights = []
    for weight, param in zip(quantized_weights, params):
        dequantized_data = weight.astype(np.float32) / 127
        denorm_data = (dequantized_data + 1) / 2 * (param["max"] - param["min"]) + param["min"]
        dequantized_weights.append(denorm_data)

    return dequantized_weights


def weights2list(weights):
    return [w.tolist() for w in weights]

def list2np(weights):
    return np.array([np.array(w) for w in weights])