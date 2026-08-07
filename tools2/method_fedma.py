import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def fedma_aggregate(global_weights, local_weights_list):
    """
    Agregacja FedMA (Federated Matched Averaging) odporna na warstwy Flatten oraz rozgałęzienia DAG.
    Obsługuje zarówno sieci 1D (Conv1D dla IoT / szeregów czasowych),
    jak i 2D (Conv2D dla obrazów) oraz warstwy gęste (Dense).
    """
    new_global_weights = []
    num_clients = len(local_weights_list)

    prev_input_perm = [None] * num_clients
    current_output_perm = [None] * num_clients

    for layer_idx in range(len(global_weights)):
        layer_shape = global_weights[layer_idx].shape
        is_bias = len(layer_shape) == 1
        is_dense = len(layer_shape) == 2
        is_conv1d = len(layer_shape) == 3  # <-- NOWE: Obsługa Conv1D (IoT / szeregi czasowe)
        is_conv2d = len(layer_shape) == 4  # <-- Obsługa Conv2D

        is_last_layer_weights = (layer_idx >= len(global_weights) - 2)
        matched_local_layers = []

        for client_idx, local_w in enumerate(local_weights_list):
            client_layer = local_w[layer_idx].copy()

            # --- A. OBSŁUGA BIASU ---
            if is_bias:
                perm = current_output_perm[client_idx]
                if perm is not None and len(perm) == len(client_layer):
                    client_layer = client_layer[perm]
                matched_local_layers.append(client_layer)
                continue

            # --- B. OBSŁUGA WAG (Wyrównanie Wejścia / Permutacja wejściowa) ---
            in_perm = prev_input_perm[client_idx]

            if in_perm is not None:
                if is_conv1d:
                    # Kształt Conv1D: [kernel_size, in_channels, out_channels] -> Kanały wejściowe to oś 1
                    if len(in_perm) == client_layer.shape[1]:
                        client_layer = client_layer[:, in_perm, :]
                elif is_conv2d:
                    # Kształt Conv2D: [kernel_h, kernel_w, in_channels, out_channels] -> Kanały wejściowe to oś 2
                    if len(in_perm) == client_layer.shape[2]:
                        client_layer = client_layer[:, :, in_perm, :]
                elif is_dense:
                    dim_in = client_layer.shape[0]
                    if len(in_perm) == dim_in:
                        # Klasyczne połączenie 1:1 (np. Dense -> Dense)
                        client_layer = client_layer[in_perm, :]
                    elif dim_in % len(in_perm) == 0:
                        # Wykryto spłaszczenie (Conv1D/Conv2D -> Flatten -> Dense)
                        C = len(in_perm)
                        HW = dim_in // C
                        Out = client_layer.shape[1]

                        # Reshape: odtworzenie wymiaru przestrzennego/czasowego, permutacja kanałów i ponowne spłaszczenie
                        reshaped = client_layer.reshape(HW, C, Out)
                        permuted = reshaped[:, in_perm, :]
                        client_layer = permuted.reshape(-1, Out)

            # --- C. SZUKANIE PERMUTACJI (Wyrównanie Wyjścia za pomocą algorytmu węgierskiego) ---
            if is_last_layer_weights:
                current_output_perm[client_idx] = None
            else:
                if is_conv1d or is_conv2d:
                    num_filters = layer_shape[-1]
                    g_flat = global_weights[layer_idx].reshape(-1, num_filters).T
                    c_flat = client_layer.reshape(-1, num_filters).T
                elif is_dense:
                    g_flat = global_weights[layer_idx].T
                    c_flat = client_layer.T
                else:
                    # Bezpieczny fallback dla niestandardowych warstw
                    g_flat = global_weights[layer_idx].reshape(-1, layer_shape[-1]).T
                    c_flat = client_layer.reshape(-1, layer_shape[-1]).T

                cost_matrix = cdist(g_flat, c_flat, metric='euclidean')
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                current_output_perm[client_idx] = col_ind

                if is_conv1d:
                    client_layer = client_layer[:, :, col_ind]
                elif is_conv2d:
                    client_layer = client_layer[:, :, :, col_ind]
                elif is_dense:
                    client_layer = client_layer[:, col_ind]

            matched_local_layers.append(client_layer)

        if not is_bias:
            prev_input_perm = current_output_perm[:]

        avg_layer = np.mean(matched_local_layers, axis=0)
        new_global_weights.append(avg_layer)

    return new_global_weights