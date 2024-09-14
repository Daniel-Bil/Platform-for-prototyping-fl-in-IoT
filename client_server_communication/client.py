import os
import random
import socket
import json
import sys
import time
import tensorflow as tf

import numpy as np
from colorama import Fore
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Backend.magisterka import recreate_architecture_from_json2

x_train = np.random.rand(100, 3)
y_train = np.random.randint(0, 2, size=(100,))

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


def function1(data):
    print("function1")
    model = recreate_architecture_from_json2(data["data"]["data"], data["data"]["name"])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                        tf.keras.metrics.Precision(name='precision'),
                                                        tf.keras.metrics.Recall(name='recall')])
    return model

def function2(model, data):
    print("function2")
    model.set_weights(np.array([np.array(w) for w in data["weights"]]))
    history = model.fit(x_train, y_train, epochs=10)
    return history

def function2prox(model, data, dataset):
    print("function2prox")
    model.set_weights(np.array([np.array(w) for w in data["weights"]]))

    error = random.randint(4, len(dataset))
    results = {
        "loss": [],
        "accuracy": []
    }
    for idx, (batch_data, batch_labels) in enumerate(dataset):
        # Perform a training step
        if idx == error-1:
            break
        print(f"train batch nr {idx}")
        loss, accuracy, _, _ = model.train_on_batch(batch_data, batch_labels)
        results["loss"].append(loss)
        results["accuracy"].append(accuracy)

    return results, error

def function2paqfloat(model, data):
    print("function2")
    model.set_weights(model.simple_dequantize_floats(np.array([np.array(w) for w in data["weights"]])))
    history = model.fit(x_train, y_train, epochs=10)
    return history

def function2paqint(model, data):
    print("function2")

    model.set_weights(model.dequantize_weights_int(np.array([np.array(w) for w in data["weights"]]), data["params"]))
    history = model.fit(x_train, y_train, epochs=10)
    return history

def function3():
    print("f3")

def main():
    print("start client")
    batch_size = 10
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(('localhost', 8090))  # Connect to the server
        model = None

        while True:

            # Receive weights from server
            print(f"{Fore.YELLOW}Receive data{Fore.RESET}")
            received_data = receive_full_data(sock)

            data = json.loads(received_data)
            print(f"{Fore.LIGHTCYAN_EX}Received data from server:{data['id']} {data['name']} {data['method']} {data['header']}{Fore.RESET}")
            if data["header"] == "3":
                print(f"{Fore.LIGHTRED_EX} CLOSE CLIENT{Fore.RESET}")
                break

            if data["method"] == "fedavg":
                if data["header"]=="1":
                    model = function1(data)
                    history = function2(model, data)
                if data["header"]=="2":
                    history = function2(model, data)

                data_to_send = json.dumps({"weights": [w.tolist() for w in model.get_weights()],
                                           "summary": history.history})
                send_full_data(sock, data_to_send)
                print("Sent updated weights + history to server")

            if data["method"] == "fedprox":
                if data["header"] == "1":
                    model = function1(data)
                    history, error = function2prox(model, data, train_dataset)
                if data["header"] == "2":
                    history, error = function2prox(model, data, train_dataset)

                data_to_send = json.dumps({"weights": [w.tolist() for w in model.get_weights()],
                                           "summary": history,
                                           "error": error})
                send_full_data(sock, data_to_send)
                print("Sent updated weights + history to server")
            if data["method"] == "fedpaq_float":
                if data["header"] == "1":
                    model = function1(data)
                    history = function2paqfloat(model, data)
                if data["header"] == "2":
                    history = function2paqfloat(model, data)

                data_to_send = json.dumps({"weights": [w.tolist() for w in model.simple_quantize_floats(model.get_weights())],
                                           "summary": history.history})
                send_full_data(sock, data_to_send)
                print("Sent updated weights + history to server")

            if data["method"] == "fedpaq_int":
                if data["header"] == "1":
                    model = function1(data)
                    history = function2paqint(model, data)
                if data["header"] == "2":
                    history = function2paqint(model, data)
                q_weights, params = model.quantize_weights_int(model.get_weights())
                data_to_send = json.dumps({"weights": [w.tolist() for w in q_weights],
                                           "params": params,
                                           "summary": history.history})
                send_full_data(sock, data_to_send)
                print("Sent updated weights + history to server")

if __name__ == "__main__":
    main()
