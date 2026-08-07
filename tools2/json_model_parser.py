import json
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model_from_json(json_path, seq_len, num_features):
    """
    Buduje model Keras Functional API na podstawie pliku JSON z GUI.
    Ignoruje sekcję 'config', skupia się wyłącznie na 'architecture'.
    Automatycznie konwertuje warstwy Conv2D/Pool2D na 1D pod szeregi czasowe IoT.
    """
    print(f"  🔍 [JSON Parser] Wczytywanie architektury z pliku: {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = {node["id"]: node for node in data["architecture"]["nodes"]}
    edges = [edge for edge in data["architecture"]["edges"] if edge.get("isValid", True)]

    # Mapa przechowująca gotowe tensory warstw Keras
    tensor_map = {}

    # 1. Znalezienie węzła wejściowego (Input)
    input_nodes = [n for n in nodes.values() if n["type"] == "Input"]
    if not input_nodes:
        raise ValueError("Brak węzła typu 'Input' w pliku JSON!")

    input_node = input_nodes[0]
    keras_input = layers.Input(shape=(seq_len, num_features), name=input_node["id"])
    tensor_map[input_node["id"]] = keras_input

    # 2. Sortowanie topologiczne / przechodzenie grafu połączeń
    in_degree = {node_id: 0 for node_id in nodes}
    adj_list = {node_id: [] for node_id in nodes}

    for edge in edges:
        u, v = edge["from"], edge["to"]
        adj_list[u].append(v)
        in_degree[v] += 1

    queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

    while queue:
        curr_id = queue.pop(0)
        curr_node = nodes[curr_id]
        node_type = curr_node["type"]
        params = curr_node.get("params", {})

        if curr_id not in tensor_map:
            # Szukanie tensorów wejściowych od poprzedników
            incoming_edges = [e["from"] for e in edges if e["to"] == curr_id]
            input_tensors = [tensor_map[parent_id] for parent_id in incoming_edges]

            x = input_tensors[0] if len(input_tensors) == 1 else input_tensors

            # --- MAPOWANIE WARSTW KERAS Z ADAPTACJĄ DO 1D ---
            if node_type in ["Conv1D", "Conv2D"]:
                filters = params.get("filters", 32)
                kernel_size = params.get("kernel_size", 3)
                activation = params.get("activation", "relu")
                x = layers.Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, padding="same", name=curr_id)(x)

            elif node_type in ["MaxPooling1D", "MaxPooling2D"]:
                pool_size = params.get("pool_size", 2)
                x = layers.MaxPooling1D(pool_size=pool_size, padding="same", name=curr_id)(x)

            elif node_type == "Flatten":
                x = layers.Flatten(name=curr_id)(x)

            elif node_type == "Dense":
                units = params.get("units", 32)
                activation = params.get("activation", "relu")
                # Sprawdzenie, czy to ostatni węzeł w grafie (wyjście z modelu)
                if not adj_list[curr_id]:
                    print(f"    -> Wykryto warstwę wyjściową ({curr_id}). Wymuszam Dense(1, 'sigmoid') dla detekcji anomalii.")
                    x = layers.Dense(1, activation="sigmoid", name=curr_id)(x)
                else:
                    x = layers.Dense(units=units, activation=activation, name=curr_id)(x)

            elif node_type == "Dropout":
                rate = params.get("rate", 0.2)
                x = layers.Dropout(rate, name=curr_id)(x)

            elif node_type in ["LSTM", "GRU"]:
                units = params.get("units", 32)
                rnn_layer = layers.LSTM if node_type == "LSTM" else layers.GRU
                x = rnn_layer(units=units, return_sequences=bool(adj_list[curr_id]), name=curr_id)(x)

            elif node_type == "Concatenate":
                x = layers.Concatenate(name=curr_id)(x)

            elif node_type == "ReLU":
                x = layers.ReLU(name=curr_id)(x)

            else:
                print(f"    ⚠️ Nieznana warstwa {node_type} ({curr_id}) -> Pomijam/Przepuszczam sygnał.")

            tensor_map[curr_id] = x

        for neighbor in adj_list[curr_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # 3. Zbudowanie gotowego modelu Keras Functional API
    output_nodes = [node_id for node_id, neighbors in adj_list.items() if len(neighbors) == 0]
    output_tensor = tensor_map[output_nodes[0]]

    model = models.Model(inputs=keras_input, outputs=output_tensor, name="JSON_Federated_Model")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print(f"  ✅ [JSON Parser] Pomyślnie zbudowano model: {len(model.layers)} warstw.")
    return model