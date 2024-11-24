import os
import json
import sys
import asyncio
import time

import numpy as np
import pandas as pd
from colorama import Fore
from sklearn.model_selection import train_test_split

import tensorflow as tf
from pathlib import Path
from scipy.optimize import linear_sum_assignment


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from client_server_communication.communication import send_full_data, receive_full_data
from client_server_communication.converting import weights2list, list2np, simple_quantize_floats, \
    simple_dequantize_floats, quantize_weights_int, dequantize_weights_int
from Backend.magisterka import recreate_architecture_from_json2

lock = asyncio.Lock()


def match_and_average_layer(client_weights, layer_idx):
    """
    Match and average neurons across clients for a specific layer.

    Parameters:
    - client_weights: List of weights from all clients.
    - layer_idx: Index of the current layer to process.

    Returns:
    - Averaged weights for the matched neurons for the layer.
    """
    # Get the weights of the current layer for all clients
    layer_weights = [weights[layer_idx] for weights in client_weights]

    # Check if the layer is 1D or 2D
    if len(layer_weights[0].shape) == 1:  # Handle 1D arrays (like biases)
        num_neurons = layer_weights[0].shape[0]

        # Initialize the cost matrix for matching neurons across clients
        cost_matrix = np.zeros((num_neurons, num_neurons))

        # Fill the cost matrix by calculating the distance between neurons across clients
        for i in range(num_neurons):
            for j in range(num_neurons):
                # Calculate distance between the first client's neuron and the others
                cost_matrix[i, j] = np.linalg.norm(layer_weights[0][i] - layer_weights[1][j])

    else:  # Handle 2D arrays (like weights in Dense layers)
        num_neurons = layer_weights[0].shape[-1]

        # Initialize the cost matrix for matching neurons across clients
        cost_matrix = np.zeros((num_neurons, num_neurons))

        # Fill the cost matrix by calculating the distance between neurons across clients
        for i in range(num_neurons):
            for j in range(num_neurons):
                # Calculate distance between the first client's neuron and the others
                cost_matrix[i, j] = np.linalg.norm(layer_weights[0][:, i] - layer_weights[1][:, j])

    # Use Hungarian algorithm (linear sum assignment) to match neurons/channels
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create an array to store the averaged neurons after matching
    averaged_layer_weights = np.zeros_like(layer_weights[0])

    # Perform the matching and averaging across all clients
    if len(layer_weights[0].shape) == 1:  # For 1D layers
        for i, j in zip(row_ind, col_ind):
            averaged_layer_weights[i] = np.mean([client_weights[c][layer_idx][j] for c in range(len(client_weights))],
                                                axis=0)
    else:  # For 2D layers
        for i, j in zip(row_ind, col_ind):
            averaged_layer_weights[:, i] = np.mean(
                [client_weights[c][layer_idx][:, j] for c in range(len(client_weights))], axis=0)

    return averaged_layer_weights


def aggregate_weights_fedma(client_weights):
    """
    Aggregate model weights from different clients using neuron matching and averaging.

    Parameters:
    - client_weights: List of weights from all clients [weights1, weights2, ..., N].

    Returns:
    - Global weights after aggregation.
    """
    # Number of layers in the model (assuming all clients have the same architecture)
    num_layers = len(client_weights[0])

    # Initialize list to hold the global model's aggregated weights
    global_weights = []

    # Loop through each layer and match/average neurons across clients
    for layer_idx in range(num_layers):
        # Match and average the neurons for this layer
        averaged_layer = match_and_average_layer(client_weights, layer_idx)
        global_weights.append(averaged_layer)

    return global_weights


def write_to_tensorboard(history_data, log_dir, start_step=0):
    # Create a TensorBoard summary writer
    writer = tf.summary.create_file_writer(log_dir)

    # Use the summary writer to log metrics, incrementing the step for each iteration
    with writer.as_default():
        for step, (acc, loss) in enumerate(zip(history_data['accuracy'], history_data['loss']), start=start_step):
            tf.summary.scalar('accuracy', acc, step=step)
            tf.summary.scalar('loss', loss, step=step)
            writer.flush()




def load_architectures():
    print("load_architectures")
    paths = os.listdir(f"{os.getcwd()}\\..\\Backend\\architectureJsons")
    datas = []
    for idx, path in enumerate(paths):
        if idx<2:
            with open(f"{os.getcwd()}\\..\\Backend\\architectureJsons\\{path}", 'r') as file:
                data = json.load(file)[0]
                datas.append({"name": path.split(".")[0], "data": data})

    return datas

def load_methods():
    # return ["fedpaq_int", "fedpaq_float", "fedavg", "fedprox"]
    # return ["fedpaq_int", "fedavg", "fedpaq_float", "fedprox", "fedma"]
    return ["fedavg"]


async def start_lock(shared_state):
    while shared_state['global_weights'] is None:
        print(f"{Fore.LIGHTMAGENTA_EX}SHORT W8 FOR GLOBAL WEIGHTS{Fore.RESET}")
        await asyncio.sleep(0.1)  # Small delay to avoid busy-waiting


async def handle_client(reader, writer, shared_state):
    _, client_id = writer.get_extra_info('peername')
    shared_state["client_ids"].append(client_id)

    current_architecture_name = None
    print(f"{Fore.CYAN}INFO{writer.get_extra_info('peername')}{Fore.RESET}")

    await start_lock(shared_state)

    while True:

        if current_architecture_name is not None:
            while current_architecture_name == shared_state["current_model_name"] and not shared_state["finished"]:
                print(f"{Fore.LIGHTMAGENTA_EX}still the same architecture{Fore.RESET}")
                await asyncio.sleep(0.5)

        current_architecture_name = shared_state["current_model_name"]


        if shared_state["finished"]:
            print("END OF WHILE LOOP")
            await send_full_data(writer, json.dumps({"header": "3",
                                                     "name": shared_state["current_model_name"],
                                                     "method": "finished",
                                                     "id": client_id}))
            break



        for method_id, method in enumerate(shared_state['methods']):
            for iteration in range(shared_state['iterations']):

                print(Fore.LIGHTWHITE_EX,f"len {method} {iteration} = ", len(shared_state['completed_clients']),shared_state['completed_clients'], shared_state['completed_average_clients'],Fore.RESET)
                # fedavg--------------------------------------------------------------------------------------------------------------
                if method == "fedavg":
                    # Send data to the client
                    if iteration == 0:
                        data_to_send = json.dumps({
                            "header": "1",
                            "name": shared_state["current_model_name"],
                            "method": method,
                            "data": shared_state['current_architecture'],
                            "weights": weights2list(shared_state['global_weights']),
                            "id": client_id
                        })
                    else:
                        data_to_send = json.dumps({
                            "header": "2",
                            "name": shared_state["current_model_name"],
                            "method": method,
                            "weights": weights2list(shared_state['averaged_weights']),
                            "id": client_id
                        })

                    await send_full_data(writer, data_to_send)

                    # Wait for the client's updated weights
                    received_data = await receive_full_data(reader)
                    received_data_json = json.loads(received_data)
                    print(f"{Fore.LIGHTGREEN_EX}Received updated weights and history from client {client_id}{Fore.RESET}")

                    # Store received weights for this client
                    shared_state['weights'][client_id] = list2np(received_data_json["weights"])

                    # async with lock:
                    #     print(f"{Fore.LIGHTMAGENTA_EX} Write to results {Fore.RESET}")
                    #     RESULTS_DIR = "results"
                    #     result_path = os.path.join(RESULTS_DIR, f"{shared_state['current_model_name']}_{method}_{client_id}")
                    #     os.makedirs(result_path, exist_ok=True)
                    #     history_file = os.path.join(result_path, "training_history.json")
                    #     with open(history_file, 'w') as f:
                    #         json.dump(received_data_json["summary"], f)
                    #
                    #     write_to_tensorboard(received_data_json["summary"], result_path)

                    async with lock:
                        shared_state['completed_clients'].append(client_id)

                    # Wait for all clients to finish this iteration
                    while len(shared_state['completed_clients']) < shared_state['total_clients']:
                        print(f"c waiting {client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_clients']}")
                        await asyncio.sleep(0.1)  # Small delay to avoid busy-waiting

                    print(f"{Fore.LIGHTBLUE_EX} all clients received data : completed clients {shared_state['completed_clients']} {Fore.RESET}")
                    # Aggregate the weights after all clients have completed


                    # aggreagate model and evaluate
                    if client_id == shared_state["client_ids"][0]:
                        print(f"start averaging {client_id}")
                        weights = []
                        for client_id2 in shared_state['completed_clients']:
                            weights.append(list2np(shared_state['weights'][client_id2]))
                        shared_state['averaged_weights'] = np.mean(weights, axis=0)
                        print("averaging done")

                        test_model = recreate_architecture_from_json2(shared_state['current_architecture']["data"], shared_state['current_architecture']["name"])
                        test_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                                                        tf.keras.metrics.Precision(name='precision'),
                                                                                        tf.keras.metrics.Recall(name='recall')])

                        results = test_model.evaluate(shared_state['test_data_x'], shared_state['test_data_y'])
                        loss = results[0]
                        accuracy = results[1]

                        results_dir = Path("results")
                        result_path = results_dir / f"aggregated_{shared_state['current_model_name']}_{method}"
                        result_path.mkdir(parents=True, exist_ok=True)
                        evaluation_file = result_path / "aggregated_evaluation.json"
                        with evaluation_file.open('w') as f:
                            json.dump({"loss": loss, "accuracy": accuracy}, f)

                        write_to_tensorboard({"accuracy": [accuracy], "loss": [loss]}, str(result_path), start_step=iteration)

                    async with lock:
                        print(f"{Fore.LIGHTBLUE_EX}append average with lock{Fore.RESET}")
                        shared_state['completed_average_clients'].append(client_id)

                    while len(shared_state['completed_average_clients']) < shared_state['total_clients']:
                        print(f"a waiting {client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_clients']} ")
                        await asyncio.sleep(0.1)


                    async with lock:
                        shared_state['completed_clients'].remove(client_id)
                        print(f"{Fore.LIGHTWHITE_EX} Removed {client_id} -> {shared_state['completed_clients']} completed_clients")

                    while len(shared_state['completed_clients']) > 0:
                        print(f"waiting {client_id} for others to remove itself com:{shared_state['completed_clients']}")
                        await asyncio.sleep(0.1)

                    async with lock:
                        shared_state['completed_average_clients'].remove(client_id)
                        print(f"{Fore.LIGHTWHITE_EX} Removed {client_id} -> {shared_state['completed_average_clients']} completed average")

                    while len(shared_state['completed_average_clients']) > 0:
                        print(f"waiting {client_id} for others to remove itself com:{shared_state['completed_average_clients']}")
                        await asyncio.sleep(0.1)

                    print(f"end iteration for {client_id}")
                    print(f"{client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_clients']}")
                    if iteration+1 == shared_state["iterations"]:
                        time.sleep(1)

                    if iteration+1 == shared_state["iterations"] and method_id+1 == len(shared_state['methods']):
                        async with lock:
                            shared_state['completed_architecture'].append(client_id)
                # fedprox--------------------------------------------------------------------------------------------------------------
                if method == "fedprox":
                    # Send data to the client
                    if iteration == 0:
                        data_to_send = json.dumps({
                            "header": "1",
                            "name": shared_state["current_model_name"],
                            "method": method,
                            "data": shared_state['current_architecture'],
                            "weights": weights2list(shared_state['global_weights']),
                            "id": client_id
                        })
                    else:
                        data_to_send = json.dumps({
                            "header": "2",
                            "name": shared_state["current_model_name"],
                            "method": method,
                            "weights": weights2list(shared_state['averaged_weights']),
                            "id": client_id
                        })

                    await send_full_data(writer, data_to_send)

                    # Wait for the client's updated weights
                    received_data = await receive_full_data(reader)
                    received_data_json = json.loads(received_data)
                    print(f"{Fore.LIGHTGREEN_EX}Received updated weights and history from client {client_id}{Fore.RESET}")

                    # Store received weights for this client
                    shared_state['weights'][client_id] = list2np(received_data_json["weights"])
                    shared_state['errors'][client_id] = received_data_json["error"]

                    # async with lock:
                    #     print(f"{Fore.LIGHTMAGENTA_EX} Write to results {Fore.RESET}")
                    #     RESULTS_DIR = "results"
                    #     result_path = os.path.join(RESULTS_DIR, f"{shared_state['current_model_name']}_{method}_{client_id}")
                    #     os.makedirs(result_path, exist_ok=True)
                    #     history_file = os.path.join(result_path, "training_history.json")
                    #     with open(history_file, 'w') as f:
                    #         json.dump(received_data_json["summary"], f)
                    #
                    #     write_to_tensorboard(received_data_json["summary"], result_path)

                    async with lock:
                        shared_state['completed_clients'].append(client_id)

                    # Wait for all clients to finish this iteration
                    while len(shared_state['completed_clients']) < shared_state['total_clients']:
                        print(f"c waiting {client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_clients']}")
                        await asyncio.sleep(0.1)  # Small delay to avoid busy-waiting

                    print(f"{Fore.LIGHTBLUE_EX} all clients received data : completed clients {shared_state['completed_clients']} {Fore.RESET}")
                    # Aggregate the weights after all clients have completed



                    if client_id == shared_state["client_ids"][0]:
                        print(f"start averaging {client_id}")
                        weights = []
                        errors = []
                        for client_id2 in shared_state['completed_clients']:

                            error_scaled_wieghts = [w * shared_state["errors"][client_id2] for w in shared_state['weights'][client_id2]]
                            weights.append(error_scaled_wieghts)
                            errors.append(shared_state["errors"][client_id2])
                        avg_weights = np.sum(weights, axis=0) / sum(errors)
                        shared_state['averaged_weights'] = avg_weights
                        print("averaging done")

                        test_model = recreate_architecture_from_json2(shared_state['current_architecture']["data"],
                                                                      shared_state['current_architecture']["name"])
                        test_model.compile(optimizer='adam', loss='binary_crossentropy',
                                           metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                    tf.keras.metrics.Precision(name='precision'),
                                                    tf.keras.metrics.Recall(name='recall')])

                        results = test_model.evaluate(shared_state['test_data_x'], shared_state['test_data_y'])
                        loss = results[0]
                        accuracy = results[1]

                        results_dir = Path("results")
                        result_path = results_dir / f"aggregated_{shared_state['current_model_name']}_{method}"
                        result_path.mkdir(parents=True, exist_ok=True)
                        evaluation_file = result_path / "aggregated_evaluation.json"
                        with evaluation_file.open('w') as f:
                            json.dump({"loss": loss, "accuracy": accuracy}, f)

                        write_to_tensorboard({"accuracy": [accuracy], "loss": [loss]}, str(result_path),
                                             start_step=iteration)


                    async with lock:
                        print(f"{Fore.LIGHTBLUE_EX}append average with lock{Fore.RESET}")
                        shared_state['completed_average_clients'].append(client_id)

                    while len(shared_state['completed_average_clients']) < shared_state['total_clients']:
                        print(f"a waiting {client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_clients']} ")
                        await asyncio.sleep(0.1)


                    async with lock:
                        shared_state['completed_clients'].remove(client_id)
                        print(f"{Fore.LIGHTWHITE_EX} Removed {client_id} -> {shared_state['completed_clients']} completed_clients")

                    while len(shared_state['completed_clients']) > 0:
                        print(f"waiting {client_id} for others to remove itself com:{shared_state['completed_clients']}")
                        await asyncio.sleep(0.1)

                    async with lock:
                        shared_state['completed_average_clients'].remove(client_id)
                        print(f"{Fore.LIGHTWHITE_EX} Removed {client_id} -> {shared_state['completed_average_clients']} completed average")

                    while len(shared_state['completed_average_clients']) > 0:
                        print(f"waiting {client_id} for others to remove itself com:{shared_state['completed_average_clients']}")
                        await asyncio.sleep(0.1)

                    print(f"end iteration for {client_id}")
                    print(f"{client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_clients']}")
                    if iteration+1 == shared_state["iterations"]:
                        time.sleep(1)

                    if iteration+1 == shared_state["iterations"] and method_id+1 == len(shared_state['methods']):
                        async with lock:
                            shared_state['completed_architecture'].append(client_id)
                # fedpaq_float--------------------------------------------------------------------------------------------------------------
                if method == "fedpaq_float":
                    # Send data to the client
                    if iteration == 0:
                        data_to_send = json.dumps({
                            "header": "1",
                            "name": shared_state["current_model_name"],
                            "method": method,
                            "data": shared_state['current_architecture'],
                            "weights": weights2list(simple_quantize_floats(shared_state['global_weights'])),
                            "id": client_id
                        })
                    else:
                        data_to_send = json.dumps({
                            "header": "2",
                            "name": shared_state["current_model_name"],
                            "method": method,
                            "weights": weights2list(simple_quantize_floats(shared_state['averaged_weights'])),
                            "id": client_id
                        })

                    await send_full_data(writer, data_to_send)

                    # Wait for the client's updated weights
                    received_data = await receive_full_data(reader)
                    received_data_json = json.loads(received_data)
                    print(f"{Fore.LIGHTGREEN_EX}Received updated weights and history from client {client_id}{Fore.RESET}")

                    # Store received weights for this client
                    shared_state['weights'][client_id] = simple_dequantize_floats(list2np(received_data_json["weights"]))

                    # async with lock:
                    #     print(f"{Fore.LIGHTMAGENTA_EX} Write to results {Fore.RESET}")
                    #     RESULTS_DIR = "results"
                    #     result_path = os.path.join(RESULTS_DIR, f"{shared_state['current_model_name']}_{method}_{client_id}")
                    #     os.makedirs(result_path, exist_ok=True)
                    #     history_file = os.path.join(result_path, "training_history.json")
                    #     with open(history_file, 'w') as f:
                    #         json.dump(received_data_json["summary"], f)
                    #
                    #     write_to_tensorboard(received_data_json["summary"], result_path)

                    async with lock:
                        shared_state['completed_clients'].append(client_id)

                    # Wait for all clients to finish this iteration
                    while len(shared_state['completed_clients']) < shared_state['total_clients']:
                        print(f"c waiting {client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_clients']}")
                        await asyncio.sleep(0.1)  # Small delay to avoid busy-waiting

                    print(f"{Fore.LIGHTBLUE_EX} all clients received data : completed clients {shared_state['completed_clients']} {Fore.RESET}")
                    # Aggregate the weights after all clients have completed



                    if client_id == shared_state["client_ids"][0]:
                        print(f"start averaging {client_id}")
                        weights = []
                        for client_id2 in shared_state['completed_clients']:
                            weights.append(shared_state['weights'][client_id2])
                        shared_state['averaged_weights'] = np.mean(weights, axis=0)
                        print("averaging done")
                        test_model = recreate_architecture_from_json2(shared_state['current_architecture']["data"],
                                                                      shared_state['current_architecture']["name"])
                        test_model.compile(optimizer='adam', loss='binary_crossentropy',
                                           metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                    tf.keras.metrics.Precision(name='precision'),
                                                    tf.keras.metrics.Recall(name='recall')])

                        results = test_model.evaluate(shared_state['test_data_x'], shared_state['test_data_y'])
                        loss = results[0]
                        accuracy = results[1]

                        results_dir = Path("results")
                        result_path = results_dir / f"aggregated_{shared_state['current_model_name']}_{method}"
                        result_path.mkdir(parents=True, exist_ok=True)
                        evaluation_file = result_path / "aggregated_evaluation.json"
                        with evaluation_file.open('w') as f:
                            json.dump({"loss": loss, "accuracy": accuracy}, f)

                        write_to_tensorboard({"accuracy": [accuracy], "loss": [loss]}, str(result_path),
                                             start_step=iteration)


                    async with lock:
                        print(f"{Fore.LIGHTBLUE_EX}append average with lock{Fore.RESET}")
                        shared_state['completed_average_clients'].append(client_id)

                    while len(shared_state['completed_average_clients']) < shared_state['total_clients']:
                        print(f"a waiting {client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_clients']} ")
                        await asyncio.sleep(0.1)


                    async with lock:
                        shared_state['completed_clients'].remove(client_id)
                        print(f"{Fore.LIGHTWHITE_EX} Removed {client_id} -> {shared_state['completed_clients']} completed_clients")

                    while len(shared_state['completed_clients']) > 0:
                        print(f"waiting {client_id} for others to remove itself com:{shared_state['completed_clients']}")
                        await asyncio.sleep(0.1)

                    async with lock:
                        shared_state['completed_average_clients'].remove(client_id)
                        print(f"{Fore.LIGHTWHITE_EX} Removed {client_id} -> {shared_state['completed_average_clients']} completed average")

                    while len(shared_state['completed_average_clients']) > 0:
                        print(f"waiting {client_id} for others to remove itself com:{shared_state['completed_average_clients']}")
                        await asyncio.sleep(0.1)

                    print(f"end iteration for {client_id}")
                    print(f"{client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_clients']}")
                    if iteration+1 == shared_state["iterations"]:
                        time.sleep(1)

                    if iteration+1 == shared_state["iterations"] and method_id+1 == len(shared_state['methods']):
                        async with lock:
                            shared_state['completed_architecture'].append(client_id)
                # fedpaq_int--------------------------------------------------------------------------------------------------------------
                if method == "fedpaq_int":
                    # Send data to the client

                    if iteration == 0:
                        q_weights, params = quantize_weights_int(shared_state['global_weights'])
                        data_to_send = json.dumps({
                            "header": "1",
                            "name": shared_state["current_model_name"],
                            "method": method,
                            "data": shared_state['current_architecture'],
                            "weights": weights2list(q_weights),
                            "id": client_id,
                            "params": params
                        })
                    else:
                        q_weights, params = quantize_weights_int(shared_state['averaged_weights'])
                        data_to_send = json.dumps({
                            "header": "2",
                            "name": shared_state["current_model_name"],
                            "method": method,
                            "weights": weights2list(q_weights),
                            "id": client_id,
                            "params": params
                        })

                    await send_full_data(writer, data_to_send)

                    # Wait for the client's updated weights
                    received_data = await receive_full_data(reader)
                    received_data_json = json.loads(received_data)
                    print(f"{Fore.LIGHTGREEN_EX}Received updated weights and history from client {client_id}{Fore.RESET}")

                    # Store received weights for this client
                    shared_state['weights'][client_id] = dequantize_weights_int(list2np(received_data_json["weights"]), received_data_json["params"])

                    async with lock:
                        shared_state['completed_clients'].append(client_id)

                    # Wait for all clients to finish this iteration
                    while len(shared_state['completed_clients']) < shared_state['total_clients']:
                        print(f"c waiting {client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_clients']}")
                        await asyncio.sleep(0.1)  # Small delay to avoid busy-waiting

                    print(f"{Fore.LIGHTBLUE_EX} all clients received data : completed clients {shared_state['completed_clients']} {Fore.RESET}")
                    # Aggregate the weights after all clients have completed



                    if client_id == shared_state["client_ids"][0]:
                        print(f"start averaging {client_id}")
                        weights = []
                        for client_id2 in shared_state['completed_clients']:
                            weights.append(list2np(shared_state['weights'][client_id2]))
                        shared_state['averaged_weights'] = np.mean(weights, axis=0)
                        print("averaging done")
                        test_model = recreate_architecture_from_json2(shared_state['current_architecture']["data"],
                                                                      shared_state['current_architecture']["name"])
                        test_model.compile(optimizer='adam', loss='binary_crossentropy',
                                           metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                    tf.keras.metrics.Precision(name='precision'),
                                                    tf.keras.metrics.Recall(name='recall')])

                        results = test_model.evaluate(shared_state['test_data_x'], shared_state['test_data_y'])
                        loss = results[0]
                        accuracy = results[1]

                        results_dir = Path("results")
                        result_path = results_dir / f"aggregated_{shared_state['current_model_name']}_{method}"
                        result_path.mkdir(parents=True, exist_ok=True)
                        evaluation_file = result_path / "aggregated_evaluation.json"
                        with evaluation_file.open('w') as f:
                            json.dump({"loss": loss, "accuracy": accuracy}, f)

                        write_to_tensorboard({"accuracy": [accuracy], "loss": [loss]}, str(result_path),
                                             start_step=iteration)


                    async with lock:
                        print(f"{Fore.LIGHTBLUE_EX}append average with lock{Fore.RESET}")
                        shared_state['completed_average_clients'].append(client_id)

                    while len(shared_state['completed_average_clients']) < shared_state['total_clients']:
                        print(f"a waiting {client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_clients']} ")
                        await asyncio.sleep(0.1)


                    async with lock:
                        shared_state['completed_clients'].remove(client_id)
                        print(f"{Fore.LIGHTWHITE_EX} Removed {client_id} -> {shared_state['completed_clients']} completed_clients")

                    while len(shared_state['completed_clients']) > 0:
                        print(f"waiting {client_id} for others to remove itself com:{shared_state['completed_clients']}")
                        await asyncio.sleep(0.1)

                    async with lock:
                        shared_state['completed_average_clients'].remove(client_id)
                        print(f"{Fore.LIGHTWHITE_EX} Removed {client_id} -> {shared_state['completed_average_clients']} completed average")

                    while len(shared_state['completed_average_clients']) > 0:
                        print(f"waiting {client_id} for others to remove itself com:{shared_state['completed_average_clients']}")
                        await asyncio.sleep(0.1)

                    print(f"end iteration for {client_id}")
                    print(f"{client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_clients']}")
                    if iteration+1 == shared_state["iterations"]:
                        time.sleep(1)

                    if iteration+1 == shared_state["iterations"] and method_id+1 == len(shared_state['methods']):
                        async with lock:
                            shared_state['completed_architecture'].append(client_id)

                if method == "fedma":
                    # Send data to the client
                    if iteration == 0:
                        data_to_send = json.dumps({
                            "header": "1",
                            "name": shared_state["current_model_name"],
                            "method": method,
                            "data": shared_state['current_architecture'],
                            "weights": weights2list(shared_state['global_weights']),
                            "id": client_id
                        })
                    else:
                        data_to_send = json.dumps({
                            "header": "2",
                            "name": shared_state["current_model_name"],
                            "method": method,
                            "weights": weights2list(shared_state['averaged_weights']),
                            "id": client_id
                        })

                    await send_full_data(writer, data_to_send)

                    # Wait for the client's updated weights
                    received_data = await receive_full_data(reader)
                    received_data_json = json.loads(received_data)
                    print(f"{Fore.LIGHTGREEN_EX}Received updated weights and history from client {client_id}{Fore.RESET}")

                    # Store received weights for this client
                    shared_state['weights'][client_id] = list2np(received_data_json["weights"])

                    # async with lock:
                    #     print(f"{Fore.LIGHTMAGENTA_EX} Write to results {Fore.RESET}")
                    #     RESULTS_DIR = "results"
                    #     result_path = os.path.join(RESULTS_DIR, f"{shared_state['current_model_name']}_{method}_{client_id}")
                    #     os.makedirs(result_path, exist_ok=True)
                    #     history_file = os.path.join(result_path, "training_history.json")
                    #     with open(history_file, 'w') as f:
                    #         json.dump(received_data_json["summary"], f)
                    #
                    #     write_to_tensorboard(received_data_json["summary"], result_path)

                    async with lock:
                        shared_state['completed_clients'].append(client_id)

                    # Wait for all clients to finish this iteration
                    while len(shared_state['completed_clients']) < shared_state['total_clients']:
                        print(f"c waiting {client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_clients']}")
                        await asyncio.sleep(0.1)  # Small delay to avoid busy-waiting

                    print(f"{Fore.LIGHTBLUE_EX} all clients received data : completed clients {shared_state['completed_clients']} {Fore.RESET}")
                    # Aggregate the weights after all clients have completed



                    if client_id == shared_state["client_ids"][0]:
                        print(f"start averaging {client_id}")
                        weights = []
                        for client_id2 in shared_state['completed_clients']:
                            weights.append(list2np(shared_state['weights'][client_id2]))

                        shared_state['averaged_weights'] = aggregate_weights_fedma(weights)
                        print("averaging done")
                        test_model = recreate_architecture_from_json2(shared_state['current_architecture']["data"],
                                                                      shared_state['current_architecture']["name"])
                        test_model.compile(optimizer='adam', loss='binary_crossentropy',
                                           metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                    tf.keras.metrics.Precision(name='precision'),
                                                    tf.keras.metrics.Recall(name='recall')])

                        results = test_model.evaluate(shared_state['test_data_x'], shared_state['test_data_y'])
                        loss = results[0]
                        accuracy = results[1]

                        results_dir = Path("results")
                        result_path = results_dir / f"aggregated_{shared_state['current_model_name']}_{method}"
                        result_path.mkdir(parents=True, exist_ok=True)
                        evaluation_file = result_path / "aggregated_evaluation.json"
                        with evaluation_file.open('w') as f:
                            json.dump({"loss": loss, "accuracy": accuracy}, f)

                        write_to_tensorboard({"accuracy": [accuracy], "loss": [loss]}, str(result_path),
                                             start_step=iteration)


                    async with lock:
                        print(f"{Fore.LIGHTBLUE_EX}append average with lock{Fore.RESET}")
                        shared_state['completed_average_clients'].append(client_id)

                    while len(shared_state['completed_average_clients']) < shared_state['total_clients']:
                        print(f"a waiting {client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_clients']} ")
                        await asyncio.sleep(0.1)


                    async with lock:
                        shared_state['completed_clients'].remove(client_id)
                        print(f"{Fore.LIGHTWHITE_EX} Removed {client_id} -> {shared_state['completed_clients']} completed_clients")

                    while len(shared_state['completed_clients']) > 0:
                        print(f"waiting {client_id} for others to remove itself com:{shared_state['completed_clients']}")
                        await asyncio.sleep(0.1)

                    async with lock:
                        shared_state['completed_average_clients'].remove(client_id)
                        print(f"{Fore.LIGHTWHITE_EX} Removed {client_id} -> {shared_state['completed_average_clients']} completed average")

                    while len(shared_state['completed_average_clients']) > 0:
                        print(f"waiting {client_id} for others to remove itself com:{shared_state['completed_average_clients']}")
                        await asyncio.sleep(0.1)

                    print(f"end iteration for {client_id}")
                    print(f"{client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_clients']}")
                    if iteration+1 == shared_state["iterations"]:
                        time.sleep(1)

                    if iteration+1 == shared_state["iterations"] and method_id+1 == len(shared_state['methods']):
                        async with lock:
                            shared_state['completed_architecture'].append(client_id)
    writer.close()
    await writer.wait_closed()


def read_csv_by_index(directory, index):
    # Get a list of all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # Check if the provided index is within the range
    if index < 0 or index >= len(csv_files):
        raise IndexError("Index out of range. Please provide a valid index.")

    # Get the filename at the specified index
    selected_file = csv_files[index]

    # Create full file path
    file_path = os.path.join(directory, selected_file)

    # Read the CSV file
    df = pd.read_csv(file_path)
    print(f"Loaded file: {selected_file}")

    return df


def create_dataset(df, window_size=5):
    data = []
    labels = []

    # Determine the middle index of the window
    middle_index = window_size // 2

    # Loop through the dataframe with a sliding window
    for i in range(len(df) - window_size + 1):
        # Extract the window for each column
        temp_window = df['value_temp'].iloc[i:i + window_size].values
        hum_window = df['value_hum'].iloc[i:i + window_size].values
        acid_window = df['value_acid'].iloc[i:i + window_size].values

        # Concatenate the values to create a single input array
        input_values = np.concatenate((temp_window, hum_window, acid_window))

        # Get the label of the middle value in the window
        label = df['label'].iloc[i + middle_index]

        # Append to the dataset
        data.append(input_values)
        labels.append(label)

    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    return data, labels

async def test2(shared_state):
    while len(shared_state['completed_architecture']) < shared_state['total_clients']:
        print(f"{Fore.LIGHTMAGENTA_EX}waiting to go to next arch{Fore.RESET}")
        await asyncio.sleep(0.5)

async def main():

    loaded_data = read_csv_by_index("..\\dane\\generated_data", 0)
    data, labels = create_dataset(loaded_data)
    x_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Load architectures and methods
    loaded_architectures = load_architectures()
    loaded_methods = load_methods()

    # Shared state among all clients
    shared_state = {
        'weights': {},  # Store the weights returned by all clients
        'errors': {},
        'completed_clients': [],  # Keep track of completed clients
        'global_weights': None,  # first weights
        'averaged_weights': None,  # Store aggregated global weights
        'iterations': 10,  # Number of iterations
        'total_clients': 2,  # Total number of clients (adjust as needed)
        'methods': loaded_methods,  # Store the methods
        "current_model_name": "",
        'completed_average_clients': [],
        "client_ids": [],
        "completed_architecture": [],
        "finished": False,
        "test_data_x": X_test,
        "test_data_y": y_test
    }
    server = await asyncio.start_server(lambda r, w: handle_client(r, w, shared_state), 'localhost', 8090)
    print("async with server")
    async with server:
        # Loop over each architecture, train, and distribute
        for architecture in loaded_architectures:
            print(f"{Fore.CYAN}Training model on the server for architecture {architecture['name']}{Fore.RESET}")
            # Train the model once on the server
            shared_state["current_model_name"] = architecture["name"]

            # Recreate and compile the model for the current architecture
            model = recreate_architecture_from_json2(architecture["data"], architecture["name"])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Train the model on the server

            model.fit(x_train, y_train, epochs=2)
            loss, accuracy = model.evaluate(X_test, y_test)
            print(f"{Fore.LIGHTGREEN_EX} Test Accuracy {shared_state['current_model_name']} before : {accuracy:.2f}{Fore.RESET}")
            # Store the trained model's weights in the shared state to distribute to clients
            shared_state['global_weights'] = model.get_weights()
            shared_state['current_architecture'] = architecture  # Track the current architecture

            await test2(shared_state)
            # while len(shared_state['completed_architecture']) < shared_state['total_clients']:
            #     print("waiting to go to next arch")
            #     await asyncio.sleep(0.5)
            # Start the server and wait for client connections

            print(f"{Fore.LIGHTBLUE_EX}clear completed_architecture in server loop cause it is 1???? i think so{Fore.RESET}")
            model.set_weights(shared_state['averaged_weights'])

            loss, accuracy = model.evaluate(X_test, y_test)
            print(f"{Fore.LIGHTGREEN_EX} Test Accuracy {shared_state['current_model_name']} after : {accuracy:.2f}{Fore.RESET}")
            shared_state['completed_architecture'] = []
        shared_state["finished"] = True
        await server.serve_forever()




if __name__ == "__main__":
    asyncio.run(main())
