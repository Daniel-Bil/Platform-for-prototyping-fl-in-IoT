import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def fedma_aggregate(global_weights, local_weights_list):
    """
    Poprawiona implementacja FedMA (Federated Matched Averaging).
    Jawnie oddziela permutacje wejściowe od wyjściowych, eliminując błędy wymiarów dla wektorów Bias.
    """
    new_global_weights = []

    num_clients = len(local_weights_list)

    # Przechowujemy osobno permutacje dla WEJŚĆ i WYJŚĆ
    prev_input_perm = [None] * num_clients
    current_output_perm = [None] * num_clients

    for layer_idx in range(len(global_weights)):
        layer_shape = global_weights[layer_idx].shape
        is_bias = len(layer_shape) == 1
        is_conv = len(layer_shape) == 4
        is_dense = len(layer_shape) == 2

        # Ostatnia warstwa wag i jej bias (klasyfikator) nie mogą być permutowane na wyjściu!
        is_last_layer_weights = (layer_idx >= len(global_weights) - 2)

        matched_local_layers = []

        for client_idx, local_w in enumerate(local_weights_list):
            client_layer = local_w[layer_idx].copy()

            # --- A. OBSŁUGA BIASU ---
            if is_bias:
                # Bias dodajemy po aktywacji, więc dziedziczy permutację WYJŚCIA z poprzedniej macierzy wag
                perm = current_output_perm[client_idx]
                if perm is not None:
                    client_layer = client_layer[perm]
                matched_local_layers.append(client_layer)
                continue

            # --- B. OBSŁUGA WAG (Conv2D / Dense) ---

            # 1. Wyrównanie WEJŚCIA (na podstawie permutacji wyjść z poprzedniej warstwy)
            in_perm = prev_input_perm[client_idx]
            if in_perm is not None:
                if is_conv:
                    client_layer = client_layer[:, :, in_perm, :]
                elif is_dense:
                    client_layer = client_layer[in_perm, :]

            # 2. Dopasowanie WYJŚCIA (szukamy nowej permutacji)
            if is_last_layer_weights:
                # Nie ruszamy wyjść ostatniej warstwy
                current_output_perm[client_idx] = None
            else:
                # Macierz kosztów i Algorytm Węgierski
                if is_conv:
                    num_filters = layer_shape[-1]
                    g_flat = global_weights[layer_idx].reshape(-1, num_filters).T
                    c_flat = client_layer.reshape(-1, num_filters).T
                elif is_dense:
                    g_flat = global_weights[layer_idx].T
                    c_flat = client_layer.T

                cost_matrix = cdist(g_flat, c_flat, metric='euclidean')
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                # Zapisujemy i aplikujemy permutację wyjść
                current_output_perm[client_idx] = col_ind
                if is_conv:
                    client_layer = client_layer[:, :, :, col_ind]
                elif is_dense:
                    client_layer = client_layer[:, col_ind]

            matched_local_layers.append(client_layer)

        # Po zrównaniu wyjść wszystkich klientów dla danej warstwy,
        # nowa permutacja wyjść staje się permutacją WEJŚĆ dla kolejnej warstwy wag.
        if not is_bias:
            prev_input_perm = current_output_perm[:]

        # --- 3. UŚREDNIENIE ---
        avg_layer = np.mean(matched_local_layers, axis=0)
        new_global_weights.append(avg_layer)

    return new_global_weights


if __name__ == "__main__":
    print("🚀 Rozpoczynam drugie podejście do testów FedMA...\n")

    try:
        # Struktura globalna (symulacja sieci Conv2D -> Dense -> Dense)
        W1_g = np.random.rand(3, 3, 1, 4)
        b1_g = np.random.rand(4)
        W2_g = np.random.rand(4, 3)
        b2_g = np.random.rand(3)
        W3_g = np.random.rand(3, 2)
        b3_g = np.random.rand(2)

        global_weights = [W1_g, b1_g, W2_g, b2_g, W3_g, b3_g]

        # Tworzymy symulowane wagi dla 2 klientów
        local_weights_list = [
            [np.random.rand(3, 3, 1, 4), np.random.rand(4),
             np.random.rand(4, 3), np.random.rand(3),
             np.random.rand(3, 2), np.random.rand(2)],

            [np.random.rand(3, 3, 1, 4), np.random.rand(4),
             np.random.rand(4, 3), np.random.rand(3),
             np.random.rand(3, 2), np.random.rand(2)]
        ]

        print("✅ Generowanie danych wejściowych zakończone pomyślnie.")

        result_weights = fedma_aggregate(global_weights, local_weights_list)

        print("\n✅ Algorytm FedMA wykonał się bez rzucania błędów!")
        print("-" * 50)

        # Weryfikacja wymiarów
        wszystko_ok = True
        for idx, w in enumerate(result_weights):
            expected_shape = global_weights[idx].shape
            actual_shape = w.shape
            status = "OK" if expected_shape == actual_shape else "BŁĄD!"
            print(f"Warstwa {idx} - Wymagany shape: {expected_shape} | Wynikowy: {actual_shape} -> [{status}]")
            if expected_shape != actual_shape:
                wszystko_ok = False

        if wszystko_ok:
            print("\n🎉 TEST ZAKOŃCZONY SUKCESEM: Matematyka tensorów jest spójna!")
        else:
            print("\n❌ TEST OBLANY: Niezgodność wymiarów!")

    except Exception as e:
        import traceback

        print(f"\n❌ BŁĄD WYKONANIA (CRASH):\n{traceback.format_exc()}")