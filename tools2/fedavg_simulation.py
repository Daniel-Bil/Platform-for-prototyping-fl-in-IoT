import json
import os
import time
import urllib.request
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from method_fedma import fedma_aggregate

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===================================================================
# 🛠️ KONFIGURACJA BENCHMARKU (TUTAJ WPISZ SWOJE WYEKSPORTOWANE PLIKI)
# ===================================================================
ARCHITECTURES = [
    "fl_model_all4.json",  # Wyeksportuj z UI i zmień nazwę na tę
    "roz_all4.json"  # Wyeksportuj inną z UI i zmień nazwę
]
CSV_OUTPUT_PATH = os.path.join(CURRENT_DIR, "wyniki_magisterka.csv")
# ALGORITHMS_TO_RUN = ['fedavg', 'fedprox', 'fedpaq', 'hierfavg', 'fedma']
ALGORITHMS_TO_RUN = ['fedma']


def send_log_to_frontend(arch, algo, round_num, acc, loss, transfer_mb, time_sec):
    url = "http://localhost:8000/api/internal/fl-log"
    data = json.dumps({
        "type": "fl_log",
        "architecture": arch,  # <--- NOWE POLE
        "algorithm": algo,
        "round": round_num,
        "accuracy": float(acc),
        "loss": float(loss),
        "transfer_mb": float(transfer_mb),
        "time_sec": float(time_sec)
    }).encode('utf-8')
    try:
        urllib.request.urlopen(urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}))
    except Exception:
        pass


def quantize_weights(weights_list, bits=8):
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


def build_model_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    nodes = {n['id']: n for n in data['architecture']['nodes']}
    edges = data['architecture']['edges']
    node_inputs = {n_id: [] for n_id in nodes}
    for edge in edges: node_inputs[edge['to']].append(edge['from'])
    tensors = {}

    def get_tensor(node_id):
        if node_id in tensors: return tensors[node_id]
        node, in_nodes = nodes[node_id], node_inputs[node_id]
        in_tensors = [get_tensor(n) for n in in_nodes]
        l_type, params = node['type'], node['params']

        if l_type == 'Input':
            t = tf.keras.Input(shape=tuple(int(x) for x in params['shape'].split(',')), name=node_id)
        elif l_type == 'Dense':
            t = layers.Dense(int(params['units']), activation=params.get('activation', 'linear'), name=node_id)(
                in_tensors[0] if in_tensors else None)
        elif l_type == 'Conv2D':
            t = layers.Conv2D(int(params['filters']), int(params['kernel_size']),
                              activation=params.get('activation', 'relu'), name=node_id)(in_tensors[0])
        elif l_type == 'MaxPooling2D':
            t = layers.MaxPooling2D(int(params['pool_size']), name=node_id)(in_tensors[0])
        elif l_type == 'Flatten':
            t = layers.Flatten(name=node_id)(in_tensors[0])
        elif l_type == 'Dropout':
            t = layers.Dropout(float(params['rate']), name=node_id)(in_tensors[0])
        elif l_type == 'Concatenate':
            t = layers.Concatenate(name=node_id)(in_tensors)
        else:
            t = in_tensors[0]
        tensors[node_id] = t
        return t

    out_nodes = set(nodes.keys()) - set([e['from'] for e in edges])
    model = models.Model(inputs=[get_tensor(n) for n in nodes if nodes[n]['type'] == 'Input'],
                         outputs=[get_tensor(n) for n in out_nodes])
    return model, data['config']


def run_benchmark():
    # --- PRZYGOTOWANIE PLIKU CSV ---
    with open(CSV_OUTPUT_PATH, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Architektura', 'Algorytm', 'Runda', 'Accuracy', 'Loss', 'Transfer_MB', 'Czas_s'])

        # --- GŁÓWNA PĘTLA PO ARCHITEKTURACH ---
        for arch_file in ARCHITECTURES:
            json_path = os.path.join(CURRENT_DIR, arch_file)
            if not os.path.exists(json_path):
                print(f"⚠️ Pominięto {arch_file} - plik nie istnieje w folderze tools/.")
                continue

            print(f"\n=======================================================")
            print(f" 🏗️ ŁADOWANIE ARCHITEKTURY: {arch_file}")
            print(f"=======================================================")
            base_model, config = build_model_from_json(json_path)
            fresh_initial_weights = base_model.get_weights()  # Punkt startowy (identyczny dla każdego algorytmu)
            model_size_mb = sum(w.nbytes for w in fresh_initial_weights) / (1024 * 1024)

            # Pobieramy parametry z JSON
            num_clients, fraction = int(config.get('clients', 100)), float(config.get('fraction', 0.1))
            local_epochs, batch_size = int(config.get('epochs', 5)), int(config.get('batch', 32))
            global_rounds = 5

            # ------------------------------------------------------------------
            # 📊 TUTAJ W PRZYSZŁOŚCI WEPIESZ DANE Z CZUJNIKÓW OD PROMOTORA
            # ------------------------------------------------------------------
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
            x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0
            client_data_x, client_data_y = np.array_split(x_train, num_clients), np.array_split(y_train, num_clients)
            # ------------------------------------------------------------------

            # --- WEWNĘTRZNA PĘTLA PO ALGORYTMACH ---
            for algorithm in ALGORITHMS_TO_RUN:
                print(f"\n---> START SYMULACJI: {algorithm.upper()} na architekturze {arch_file}")

                # Resetujemy model do wspólnego stanu początkowego
                global_model = models.clone_model(base_model)
                global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                global_model.set_weights(fresh_initial_weights)

                for round_num in range(1, global_rounds + 1):
                    round_start_time = time.time()
                    global_weights = global_model.get_weights()

                    num_selected = max(1, int(num_clients * fraction))
                    selected_clients = np.random.choice(range(num_clients), num_selected, replace=False)

                    if algorithm == 'hierfavg':
                        num_edges, kappa_2 = 2, 2
                        edge_client_groups = np.array_split(selected_clients, num_edges)
                        edge_weights_list = []

                        for edge_clients in edge_client_groups:
                            if len(edge_clients) == 0: continue
                            edge_weights = [np.copy(w) for w in global_weights]

                            for edge_round in range(kappa_2):
                                local_weights_for_edge = []
                                for client_id in edge_clients:
                                    local_model = models.clone_model(global_model)
                                    local_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                                                        metrics=['accuracy'])
                                    local_model.set_weights(edge_weights)
                                    local_model.fit(client_data_x[client_id], client_data_y[client_id],
                                                    epochs=local_epochs, batch_size=batch_size, verbose=0)
                                    local_weights_for_edge.append(local_model.get_weights())

                                edge_weights = [np.mean(w_tuple, axis=0) for w_tuple in zip(*local_weights_for_edge)]
                            edge_weights_list.append(edge_weights)

                        if algorithm == 'fedma':
                            # Wywołanie naszej zewnętrznej funkcji dopasowującej
                            new_global_weights = fedma_aggregate(global_weights, local_weights_list)
                        else:
                            # Zwykłe uśrednianie (coordinate-wise) dla FedAvg, FedProx i FedPAQ
                            new_global_weights = [np.mean(w_tuple, axis=0) for w_tuple in zip(*local_weights_list)]

                        global_model.set_weights(new_global_weights)

                    else:
                        local_weights_list = []
                        for client_id in selected_clients:
                            local_model = models.clone_model(global_model)
                            local_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                                                metrics=['accuracy'])
                            local_model.set_weights(global_weights)

                            if algorithm == 'fedprox':
                                mu = 0.01
                                optimizer, loss_fn = tf.keras.optimizers.Adam(), tf.keras.losses.SparseCategoricalCrossentropy()
                                train_dataset = tf.data.Dataset.from_tensor_slices(
                                    (client_data_x[client_id], client_data_y[client_id])).batch(batch_size)

                                for epoch in range(local_epochs):
                                    for x_batch, y_batch in train_dataset:
                                        with tf.GradientTape() as tape:
                                            logits = local_model(x_batch, training=True)
                                            base_loss = loss_fn(y_batch, logits)
                                            proximal_term = sum(tf.reduce_sum(tf.square(lw - gw)) for lw, gw in
                                                                zip(local_model.trainable_weights, global_weights))
                                            total_loss = base_loss + (mu / 2.0) * proximal_term
                                        grads = tape.gradient(total_loss, local_model.trainable_weights)
                                        optimizer.apply_gradients(zip(grads, local_model.trainable_weights))
                            else:
                                local_model.fit(client_data_x[client_id], client_data_y[client_id], epochs=local_epochs,
                                                batch_size=batch_size, verbose=0)

                            client_weights = local_model.get_weights()
                            if algorithm == 'fedpaq': client_weights = quantize_weights(client_weights, bits=8)
                            local_weights_list.append(client_weights)

                        if algorithm == 'fedma':
                            # Wywołanie naszej zewnętrznej funkcji dopasowującej
                            new_global_weights = fedma_aggregate(global_weights, local_weights_list)
                        else:
                            # Zwykłe uśrednianie (coordinate-wise) dla FedAvg, FedProx i FedPAQ
                            new_global_weights = [np.mean(w_tuple, axis=0) for w_tuple in zip(*local_weights_list)]

                        global_model.set_weights(new_global_weights)

                    # Pomiary i zapis
                    loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
                    round_time = time.time() - round_start_time

                    if algorithm == 'fedpaq':
                        data_transfer_mb = (model_size_mb + (model_size_mb / 4)) * num_selected
                    elif algorithm == 'hierfavg':
                        data_transfer_mb = ((model_size_mb * 2) * num_edges) + ((model_size_mb * 2) * num_selected * 2)
                    else:
                        data_transfer_mb = (model_size_mb * 2) * num_selected

                    print(
                        f"[{algorithm.upper()}] Runda {round_num}: Acc = {acc:.4f} | Loss = {loss:.4f} | Transfer = {data_transfer_mb:.2f} MB")

                    # 1. Wysyłamy na żywo do przeglądarki (Dodano arch_file)
                    send_log_to_frontend(arch_file, algorithm, round_num, acc, loss, data_transfer_mb, round_time)

                    # 2. Zapisujemy wiersz do pliku CSV dla Promotora
                    writer.writerow([arch_file, algorithm, round_num, acc, loss, data_transfer_mb, round_time])
                    csv_file.flush()  # Wymuszenie fizycznego zapisu po każdej rundzie


if __name__ == "__main__":
    run_benchmark()