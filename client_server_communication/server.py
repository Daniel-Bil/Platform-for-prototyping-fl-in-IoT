import os
import socket
import json
import sys

from colorama import Fore
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Backend.magisterka import recreate_architecture_from_json2

def send_full_data(sock, data, buffer_size=1024):
    # If the data is a string, encode it to bytes
    data = data.encode()

    # Send the data in chunks
    total_sent = 0
    while total_sent < len(data):
        # Send a chunk of data up to buffer_size
        sent = sock.send(data[total_sent:total_sent + buffer_size])
        if sent == 0:
            raise RuntimeError("Socket connection broken")

        # Update the total amount of data sent
        total_sent += sent

def receive_full_data(sock, buffer_size=1024):
    data = b''  # Start with empty byte string to accumulate the data
    while True:
        part = sock.recv(buffer_size)  # Receive part of the data
        data += part  # Append to the full data
        if len(part) < buffer_size:
            # If the last received part is smaller than the buffer size, it's likely the end
            break
    return data.decode()

def update_global_model(local_models):
    # Placeholder function to aggregate local models and update global model
    global_model = {"weights": [sum(model["weights"]) / len(local_models) for model in local_models]}
    return global_model



def load_architectures():
    print("load_architectures")
    paths = os.listdir(f"{os.getcwd()}\\..\\Backend\\architectureJsons")
    datas = []
    for idx, path in enumerate(paths):
        if idx<4:
            with open(f"{os.getcwd()}\\..\\Backend\\architectureJsons\\{path}", 'r') as file:
                data = json.load(file)[0]
                datas.append({"name": path.split(".")[0], "data": data})

    return datas

def load_methods():
    return ["fedpaq_int", "fedpaq_float", "fedavg", "fedprox"]
    # return ["fedavg", "fedprox", "fedpaq"]

def main():

    print("server start")
    number_of_clients = 2
    number_of_iterations = 5
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(('localhost', 8090))
        server_socket.listen(number_of_clients)

        connections = []
        for i in range(number_of_clients):
            print(f"Waiting for client {i + 1} to connect...")
            connection, client_address = server_socket.accept()
            print(f"Connected to {client_address}")
            connections.append(connection)

    print("model training and other shits")


    loaded_architectures = load_architectures()
    loaded_methods = load_methods()
    x_train = np.random.rand(100, 3)
    y_train = np.random.randint(0, 2, size=(100,))
    avg_weights = []
    for loaded_architecture in loaded_architectures:
        model = recreate_architecture_from_json2(loaded_architecture["data"], loaded_architecture["name"])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.summary()
        model.fit(x_train, y_train, epochs=40)

        print(model.predict([[2,2,2]]))
        print(model.predict([[7, 7, 7]]))
        print(model.predict([[4, 5, 4]]))
        for method in loaded_methods:
            print(f"{Fore.LIGHTBLUE_EX}      {method}{Fore.RESET}")
            for i in range(number_of_iterations):
                print()
                print(f"TEST {i}")
                print()
                if method == "fedavg":
                    if i == 0:

                        for connection in connections:
                            data_to_send = json.dumps({
                                "header": "1",
                                "name": model.name,
                                "weights": [w.tolist() for w in model.get_weights()],
                                "method": method,
                                "data": loaded_architecture,
                                "other": {}
                            })
                            send_full_data(connection, data_to_send)
                            print(f"Sent weights to client {connection.getpeername()}")
                    else:
                        for connection in connections:
                            data_to_send = json.dumps({
                                "header": "2",
                                "name": model.name,
                                "weights": [w.tolist() for w in avg_weights],
                                "method": method,
                                "data": loaded_architecture,
                                "other": {}
                            })
                            send_full_data(connection, data_to_send)
                            print(f"Sent weights to client {connection.getpeername()}")
                    weights = []
                    for connection in connections:
                        received_data = receive_full_data(connection)
                        data = json.loads(received_data)

                        weights.append(np.array([np.array(w) for w in data["weights"]]))
                    avg_weights = np.mean(weights, axis=0)
                if method == "fedprox":
                    if i == 0:

                        for connection in connections:
                            data_to_send = json.dumps({
                                "header": "1",
                                "name": model.name,
                                "weights": [w.tolist() for w in model.get_weights()],
                                "method": method,
                                "data": loaded_architecture,
                                "other": {}
                            })
                            send_full_data(connection, data_to_send)
                            print(f"Sent weights to client {connection.getpeername()}")
                    else:
                        for connection in connections:
                            data_to_send = json.dumps({
                                "header": "2",
                                "name": model.name,
                                "weights": [w.tolist() for w in avg_weights],
                                "method": method,
                                "data": loaded_architecture,
                                "other": {}
                            })
                            send_full_data(connection, data_to_send)
                            print(f"Sent weights to client {connection.getpeername()}")
                    weights = []
                    errors = []
                    for connection in connections:
                        received_data = receive_full_data(connection)
                        data = json.loads(received_data)
                        np_weights = np.array([np.array(w) for w in data["weights"]])
                        error_scaled_weights = [w * data["error"] for w in np_weights]
                        weights.append(error_scaled_weights)
                        errors.append(data["error"])
                    avg_weights = np.sum(weights, axis=0) / sum(errors)
                if method == "fedpaq_float":
                    if i == 0:

                        for connection in connections:
                            data_to_send = json.dumps({
                                "header": "1",
                                "name": model.name,
                                "weights": [w.tolist() for w in model.simple_quantize_floats(model.get_weights())] ,
                                "method": method,
                                "data": loaded_architecture,
                                "other": {}
                            })
                            send_full_data(connection, data_to_send)
                            print(f"Sent weights to client {connection.getpeername()}")
                    else:
                        for connection in connections:
                            data_to_send = json.dumps({
                                "header": "2",
                                "name": model.name,
                                "weights": [w.tolist() for w in model.simple_quantize_floats(avg_weights)],
                                "method": method,
                                "data": loaded_architecture,
                                "other": {}
                            })
                            send_full_data(connection, data_to_send)
                            print(f"Sent weights to client {connection.getpeername()}")
                    weights = []
                    for connection in connections:
                        received_data = receive_full_data(connection)
                        data = json.loads(received_data)

                        weights.append(model.simple_dequantize_floats(np.array([np.array(w) for w in data["weights"]])))
                    avg_weights = np.mean(weights, axis=0)

                if method == "fedpaq_int":
                    if i == 0:
                        q_weights, params = model.quantize_weights_int(model.get_weights())
                        listed = [w.tolist() for w in q_weights]
                        for connection in connections:
                            data_to_send = json.dumps({
                                "header": "1",
                                "name": model.name,
                                "weights": listed,
                                "params": params,
                                "method": method,
                                "data": loaded_architecture,
                                "other": {}
                            })
                            send_full_data(connection, data_to_send)
                            print(f"Sent weights to client {connection.getpeername()}")
                    else:
                        q_weights, params = model.quantize_weights_int(avg_weights)
                        for connection in connections:
                            data_to_send = json.dumps({
                                "header": "2",
                                "name": model.name,
                                "weights": [w.tolist() for w in q_weights],
                                "params": params,
                                "method": method,
                                "data": loaded_architecture,
                                "other": {}
                            })
                            send_full_data(connection, data_to_send)
                            print(f"Sent weights to client {connection.getpeername()}")
                    weights = []
                    for connection in connections:
                        received_data = receive_full_data(connection)
                        data = json.loads(received_data)

                        weights.append(model.dequantize_weights_int(np.array([np.array(w) for w in data["weights"]]),data["params"]))
                    avg_weights = np.mean(weights, axis=0)

if __name__ == "__main__":
    main()

