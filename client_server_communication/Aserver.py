import os
import json
import sys
import asyncio
import time

import numpy as np
from colorama import Fore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Backend.magisterka import recreate_architecture_from_json2
import tensorflow as tf

lock = asyncio.Lock()

def write_to_tensorboard(history_data, log_dir):
    # Create a TensorBoard summary writer
    writer = tf.summary.create_file_writer(log_dir)

    # Use the summary writer to log metrics
    with writer.as_default():
        for epoch, (acc, loss) in enumerate(zip(history_data['accuracy'], history_data['loss'])):
            tf.summary.scalar('accuracy', acc, step=epoch)
            tf.summary.scalar('loss', loss, step=epoch)

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


def load_architectures():
    print("load_architectures")
    paths = os.listdir(f"{os.getcwd()}\\..\\Backend\\architectureJsons")
    datas = []
    for idx, path in enumerate(paths):
        if idx<1:
            with open(f"{os.getcwd()}\\..\\Backend\\architectureJsons\\{path}", 'r') as file:
                data = json.load(file)[0]
                datas.append({"name": path.split(".")[0], "data": data})

    return datas

def load_methods():
    # return ["fedpaq_int", "fedpaq_float", "fedavg", "fedprox"]
    return ["fedpaq_int", "fedavg", "fedpaq_float", "fedprox"]

async def send_full_data(writer, data):
    data = data.encode()
    writer.write(data)
    await writer.drain()

async def receive_full_data(reader, buffer_size=1024):
    data = b''
    while True:
        part = await reader.read(buffer_size)
        data += part
        if len(part) < buffer_size:
            break
    return data.decode()

def weights2list(weights):
    return [w.tolist() for w in weights]

def list2np(weights):
    return np.array([np.array(w) for w in weights])

async def handle_client(reader, writer, shared_state):
    _, client_id = writer.get_extra_info('peername')
    shared_state["client_ids"].append(client_id)
    # current_architecture_name = shared_state["current_model_name"]
    current_architecture_name = None
    print(f"{Fore.CYAN}INFO{writer.get_extra_info('peername')}{Fore.RESET}")
    #wait untill server performs first action
    while shared_state['global_weights'] is None:
        await asyncio.sleep(0.1)  # Small delay to avoid busy-waiting

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

                    async with lock:
                        print(f"{Fore.LIGHTMAGENTA_EX} Write to results {Fore.RESET}")
                        RESULTS_DIR = "results"
                        result_path = os.path.join(RESULTS_DIR, f"{shared_state['current_model_name']}_{method}_{client_id}")
                        os.makedirs(result_path, exist_ok=True)
                        history_file = os.path.join(result_path, "training_history.json")
                        with open(history_file, 'w') as f:
                            json.dump(received_data_json["summary"], f)

                        write_to_tensorboard(received_data_json["summary"], result_path)

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

                    async with lock:
                        print(f"{Fore.LIGHTMAGENTA_EX} Write to results {Fore.RESET}")
                        RESULTS_DIR = "results"
                        result_path = os.path.join(RESULTS_DIR, f"{shared_state['current_model_name']}_{method}_{client_id}")
                        os.makedirs(result_path, exist_ok=True)
                        history_file = os.path.join(result_path, "training_history.json")
                        with open(history_file, 'w') as f:
                            json.dump(received_data_json["summary"], f)

                        write_to_tensorboard(received_data_json["summary"], result_path)

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

                    async with lock:
                        print(f"{Fore.LIGHTMAGENTA_EX} Write to results {Fore.RESET}")
                        RESULTS_DIR = "results"
                        result_path = os.path.join(RESULTS_DIR, f"{shared_state['current_model_name']}_{method}_{client_id}")
                        os.makedirs(result_path, exist_ok=True)
                        history_file = os.path.join(result_path, "training_history.json")
                        with open(history_file, 'w') as f:
                            json.dump(received_data_json["summary"], f)

                        write_to_tensorboard(received_data_json["summary"], result_path)

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
                        print(f"{Fore.LIGHTMAGENTA_EX} Write to results {Fore.RESET}")
                        RESULTS_DIR = "results"
                        result_path = os.path.join(RESULTS_DIR, f"{shared_state['current_model_name']}_{method}_{client_id}")
                        os.makedirs(result_path, exist_ok=True)
                        history_file = os.path.join(result_path, "training_history.json")
                        with open(history_file, 'w') as f:
                            json.dump(received_data_json["summary"], f)

                        write_to_tensorboard(received_data_json["summary"], result_path)

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

async def main():
    x_train = np.random.rand(100, 3)
    y_train = np.random.randint(0, 2, size=(100,))
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
        'iterations': 3,  # Number of iterations
        'total_clients': 2,  # Total number of clients (adjust as needed)
        'methods': loaded_methods,  # Store the methods
        "current_model_name": "",
        'completed_average_clients': [],
        "client_ids": [],
        "completed_architecture": [],
        "finished": False
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

            model.fit(x_train, y_train, epochs=40)

            # Store the trained model's weights in the shared state to distribute to clients
            shared_state['global_weights'] = model.get_weights()
            shared_state['current_architecture'] = architecture  # Track the current architecture

            while len(shared_state['completed_architecture']) < shared_state['total_clients']:
                print("waiting to go to next arch")
                await asyncio.sleep(0.5)
            # Start the server and wait for client connections
            print("clear completed_architecture in server loop cause it is 1???? i think so")
            shared_state['completed_architecture'] =[]
        shared_state["finished"] = True
        await server.serve_forever()




if __name__ == "__main__":
    asyncio.run(main())
