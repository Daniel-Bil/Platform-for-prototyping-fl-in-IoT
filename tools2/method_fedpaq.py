import numpy as np


def quantize_weights(weights_list, bits=8):
    """
    Kwantyzacja wag do podanej liczby bitów w celu zmniejszenia transferu danych.
    """
    levels = 2 ** bits - 1
    quantized_weights = []

    for w in weights_list:
        min_val, max_val = np.min(w), np.max(w)
        if max_val == min_val:
            quantized_weights.append(w)
            continue

        normalized = (w - min_val) / (max_val - min_val)
        quantized = np.round(normalized * levels)
        dequantized = (quantized / levels) * (max_val - min_val) + min_val
        quantized_weights.append(dequantized)

    return quantized_weights


def fedpaq_aggregate(local_weights_list):
    """
    Agregacja FedPAQ. Uśrednia wagi, które wcześniej zostały skwantyzowane na urządzeniach brzegowych.
    """
    return [np.mean(w_tuple, axis=0) for w_tuple in zip(*local_weights_list)]