import os
import json
import sys
import asyncio
import time

import numpy as np
from colorama import Fore
from sklearn.model_selection import train_test_split

import tensorflow as tf
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic2.Asyncio_waiters.Edge_server_waiters import wait_for_all_clients_receive_data, \
    wait_for_all_clients_aggregate_weights, wait_for_completed_removal_aggregation, wait_for_completed_removal_receive

from logic2.Federated_logic.federated_methods_server import aggregate_weights_fedma

from logic2.utilities import write_to_tensorboard, load_methods, load_architectures, read_csv_by_index, create_dataset, \
    recreate_architecture_from_json2

from logic2.Communication_logic.communication import send_full_data, receive_full_data

from logic2.weights_operations import weights2list, list2np, simple_quantize_floats, \
    simple_dequantize_floats, quantize_weights_int, dequantize_weights_int

lock = asyncio.Lock()


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
                print(Fore.LIGHTWHITE_EX, f"{method} {iteration} {client_id} {shared_state['current_model_name']}", Fore.RESET)

                # Prepare generic data_to_send
                data_to_send = {
                                    "id": client_id,
                                    "name": shared_state["current_model_name"],
                                    "method": method
                               }

                if iteration == 0:
                    data_to_send["header"] = "1"
                    data_to_send["data"] = shared_state['current_architecture']
                else:
                    data_to_send["header"] = "2"

                # add method specific keys and values to data_to_send
                if (method == "fedavg") or (method == "fedprox") or (method == "fedma"):
                    if iteration == 0:
                        data_to_send["weights"] = weights2list(shared_state['global_weights'])
                    else:
                        data_to_send["weights"] = weights2list(shared_state['averaged_weights'])

                if method == "fedpaq_int":
                    if iteration == 0:
                        q_weights, params = quantize_weights_int(shared_state['global_weights'])
                        data_to_send["weights"] = weights2list(q_weights)
                        data_to_send["params"] = params

                    else:
                        q_weights, params = quantize_weights_int(shared_state['averaged_weights'])
                        data_to_send["weights"] = weights2list(q_weights)
                        data_to_send["params"] = params

                if method == "fedpaq_float":
                    if iteration == 0:
                        data_to_send["weights"] = weights2list(simple_quantize_floats(shared_state['global_weights']))
                    else:
                        data_to_send["weights"] = weights2list(simple_quantize_floats(shared_state['averaged_weights']))


                data_to_send_json_string = json.dumps(data_to_send)
                await send_full_data(writer, data_to_send_json_string)

                # Wait for the client's updated weights
                received_data_json_string = await receive_full_data(reader)
                received_data_dict = json.loads(received_data_json_string)
                print(f"{Fore.LIGHTGREEN_EX}Received updated weights and history from client {client_id}{Fore.RESET}")

                # Unpack received data correctly
                if (method == "fedavg"):
                    async with lock:
                        # Store received weights for this client
                        shared_state['weights'][client_id] = list2np(received_data_dict["weights"])
                        shared_state['completed_receiving_clients'].append(client_id)

                if (method == "fedprox"):
                    async with lock:
                        # Store received weights for this client
                        shared_state['weights'][client_id] = list2np(received_data_dict["weights"])
                        shared_state['errors'][client_id] = received_data_dict["error"]
                        shared_state['completed_receiving_clients'].append(client_id)

                if (method == "fedma"):
                    async with lock:
                        shared_state['weights'][client_id] = list2np(received_data_dict["weights"])
                        shared_state['completed_receiving_clients'].append(client_id)

                if method == "fedpaq_int":
                    async with lock:
                        shared_state['weights'][client_id] = dequantize_weights_int(received_data_dict["weights"], received_data_dict["params"])
                        shared_state['completed_receiving_clients'].append(client_id)

                if method == "fedpaq_float":
                    async with lock:
                        shared_state['weights'][client_id] = simple_dequantize_floats(received_data_dict["weights"])
                        shared_state['completed_receiving_clients'].append(client_id)

                await wait_for_all_clients_receive_data(shared_state, client_id)




                # aggreagate model and evaluate
                if client_id == shared_state["client_ids"][0]:
                    print(f"start averaging {client_id} {method}")
                    results_path = Path("..") / "results" / f"aggregated_{shared_state['current_model_name']}_{method}"
                    results_path.mkdir(parents=True, exist_ok=True)
                    evaluation_file = results_path / "aggregated_evaluation.json"

                    weights = []

                    if (method == "fedavg"):
                        for client_id2 in shared_state['completed_receiving_clients']:
                            weights.append(list2np(shared_state['weights'][client_id2]))
                        shared_state['averaged_weights'] = np.mean(weights, axis=0)

                    if (method == "fedprox"):
                        errors = []
                        for client_id2 in shared_state['completed_receiving_clients']:
                            error_scaled_wieghts = [w * shared_state["errors"][client_id2] for w in
                                                    shared_state['weights'][client_id2]]
                            weights.append(error_scaled_wieghts)
                            errors.append(shared_state["errors"][client_id2])
                        avg_weights = np.sum(weights, axis=0) / sum(errors)
                        shared_state['averaged_weights'] = avg_weights

                    if (method == "fedma"):
                        for client_id2 in shared_state['completed_receiving_clients']:
                            weights.append(list2np(shared_state['weights'][client_id2]))

                        shared_state['averaged_weights'] = aggregate_weights_fedma(weights)

                    if method == "fedpaq_int":
                        for client_id2 in shared_state['completed_receiving_clients']:
                            weights.append(list2np(shared_state['weights'][client_id2]))
                        shared_state['averaged_weights'] = np.mean(weights, axis=0)

                    if method == "fedpaq_float":
                        for client_id2 in shared_state['completed_receiving_clients']:
                            weights.append(shared_state['weights'][client_id2])
                        shared_state['averaged_weights'] = np.mean(weights, axis=0)

                    print(f"{Fore.LIGHTCYAN_EX}averaging done {client_id}{Fore.RESET}")

                    test_model = recreate_architecture_from_json2(shared_state['current_architecture']["data"],
                                                                  shared_state['current_architecture']["name"])
                    test_model.compile(optimizer='adam', loss='binary_crossentropy',
                                       metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                tf.keras.metrics.Precision(name='precision'),
                                                tf.keras.metrics.Recall(name='recall')])

                    results = test_model.evaluate(shared_state['test_data_x'], shared_state['test_data_y'])
                    loss = results[0]
                    accuracy = results[1]

                    with evaluation_file.open('w') as f:
                        json.dump({"loss": loss, "accuracy": accuracy}, f)

                    write_to_tensorboard({"accuracy": [accuracy], "loss": [loss]}, results_path,
                                         start_step=iteration)

                async with lock:
                    print(f"{Fore.LIGHTBLUE_EX}append average{Fore.RESET}")
                    shared_state['completed_average_clients'].append(client_id)

                await wait_for_all_clients_aggregate_weights(shared_state, client_id)

                async with lock:
                    shared_state['completed_receiving_clients'].remove(client_id)
                    print(f"{Fore.LIGHTWHITE_EX} Removed {client_id} -> {shared_state['completed_receiving_clients']} completed_receiving_clients")

                await wait_for_completed_removal_receive(shared_state, client_id)

                async with lock:
                    shared_state['completed_average_clients'].remove(client_id)
                    print(f"{Fore.LIGHTWHITE_EX} Removed {client_id} -> {shared_state['completed_average_clients']} completed average")

                await wait_for_completed_removal_aggregation(shared_state, client_id)

                print(f"{Fore.LIGHTBLUE_EX}end iteration for {client_id}{Fore.RESET}")
                print(
                    f"{client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_receiving_clients']}")
                if iteration + 1 == shared_state["iterations"]:
                    time.sleep(1)

                if iteration + 1 == shared_state["iterations"] and method_id + 1 == len(shared_state['methods']):
                    async with lock:
                        shared_state['completed_architecture'].append(client_id)


    writer.close()
    await writer.wait_closed()


async def test2(shared_state):
    while len(shared_state['completed_architecture']) < shared_state['total_clients']:
        print(f"{Fore.LIGHTMAGENTA_EX}waiting to go to next arch{Fore.RESET}")
        await asyncio.sleep(0.5)


async def main():

    loaded_data = read_csv_by_index(Path("..")/"dane"/"generated_data", 0)
    data, labels = create_dataset(loaded_data)
    x_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Load architectures and methods
    loaded_architectures = load_architectures(Path("..")/"Backend"/"architectureJsons")
    loaded_methods = load_methods()

    # Shared state among all clients
    shared_state = {
        'weights': {},  # Store the weights returned by all clients
        'errors': {},
        'global_weights': None,  # first weights
        'averaged_weights': None,  # Store aggregated global weights
        'iterations': 10,  # Number of iterations
        'total_clients': 2,  # Total number of clients (adjust as needed)
        'methods': loaded_methods,  # Store the methods
        "current_model_name": "",
        'completed_average_clients': [],
        "completed_receiving_clients": [],
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

            print(f"{Fore.LIGHTBLUE_EX}clear completed_architecture in server loop cause it is 1???? i think so{Fore.RESET}")
            model.set_weights(shared_state['averaged_weights'])

            loss, accuracy = model.evaluate(X_test, y_test)
            print(f"{Fore.LIGHTGREEN_EX} Test Accuracy {shared_state['current_model_name']} after : {accuracy:.2f}{Fore.RESET}")
            shared_state['completed_architecture'] = []
        shared_state["finished"] = True
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
