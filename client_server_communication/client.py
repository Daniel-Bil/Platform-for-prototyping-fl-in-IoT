import os
import random
import socket
import json
import sys
import time
import struct
import pandas as pd
import tensorflow as tf
import argparse

import numpy as np
from colorama import Fore
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Backend.magisterka import recreate_architecture_from_json2


def convert_np2list(weights):
    if isinstance(weights, (np.ndarray, list)):
        return [convert_np2list(x) for x in weights]
    elif isinstance(weights, np.generic):
        return weights.item()
    else:
        return weights

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
    test = 0

    raw_msglen = sock.recv(4)
    if not raw_msglen:
        raise ValueError("Failed to receive data length.")
    msglen = struct.unpack('!I', raw_msglen)[0]
    print(f"I the client will receive {msglen} bytes or something ;p")
    data = b''
    while len(data) < msglen:
        part = sock.recv(buffer_size)
        print(f"acquired part of length = {len(part)}")
        if not part:
            if len(data) < msglen:
                print(f"{Fore.LIGHTRED_EX}INCOMPLETE TRANSMISION {len(data)} < {msglen} {Fore.RESET}")
            break
        data += part

    # Print the first and last 30 bytes for debugging
    print("First 30 bytes = ", data[:30])
    print("Last 30 bytes = ", data[-30:])

    return data.decode()


    # while True:
    #     part = sock.recv(buffer_size)  # Receive part of the data
    #     data += part  # Append to the full data
    #     test+=1
    #     if len(part) < buffer_size:
    #         # If the last received part is smaller than the buffer size, it's likely the end
    #         break
    # # Print the first 30 bytes
    # print(test)
    # print("First 30 bytes = ", data[:30])
    #
    # # Print the last 30 bytes
    # print("Last 30 bytes = ", data[-30:])
    # return data.decode()


def function1(data):
    print("function1")
    model = recreate_architecture_from_json2(data["data"]["data"], data["data"]["name"])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                        tf.keras.metrics.Precision(name='precision'),
                                                        tf.keras.metrics.Recall(name='recall')])
    return model

def function2(model, data, X_train, Y_train):
    print("function2")
    model.set_weights(np.array([np.array(w) for w in data["weights"]]))
    history = model.fit(X_train, Y_train, epochs=2)
    return history


def function2betterprox(model, data, dataset, mu=0.01):
    print("function2prox")

    # Get the initial global weights (from the server)
    global_weights = np.array([np.array(w) for w in data["weights"]])

    # Set the local model's weights to the global weights
    model.set_weights(global_weights)

    error = random.randint(4, len(dataset))
    results = {
        "loss": [],
        "accuracy": []
    }

    optimizer = model.optimizer

    # Perform training with the proximal term
    for idx, (batch_data, batch_labels) in enumerate(dataset):
        if idx == error - 1:
            break
        print(f"train batch nr {idx}")

        # Use GradientTape to manually calculate gradients
        with tf.GradientTape() as tape:
            # Forward pass: compute predictions and regular loss
            predictions = model(batch_data, training=True)
            loss = model.compiled_loss(batch_labels, predictions)  # Use model's compiled loss function (e.g., cross-entropy)

            # Compute the proximal term: penalty for weight deviation
            local_weights = model.trainable_variables
            prox_term = 0
            for local_w, global_w in zip(local_weights, global_weights):
                prox_term += tf.reduce_sum(tf.square(local_w - global_w))

            # Add the proximal term to the loss
            prox_penalty = (mu / 2) * prox_term
            total_loss = loss + prox_penalty

        # Compute gradients with respect to the total loss
        gradients = tape.gradient(total_loss, model.trainable_variables)

        # Apply gradients to update the model's weights
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        model.compiled_metrics.update_state(batch_labels, predictions)

        # Access the accuracy (or other metrics) results
        accuracy = model.compiled_metrics.metrics[0].result()  # Assuming

        # Save the results
        results["loss"].append(float(total_loss.numpy()))  # Convert the tensor to a numpy value
        results["accuracy"].append(float(accuracy.numpy()))

    return results, error

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

def function2paqfloat(model, data, X_train, Y_train):
    print("function2")
    model.set_weights(model.simple_dequantize_floats(np.array([np.array(w) for w in data["weights"]])))
    history = model.fit(X_train, Y_train, epochs=2)
    return history

def function2paqint(model, data, X_train, Y_train):
    print("function2")

    model.set_weights(model.dequantize_weights_int(np.array([np.array(w) for w in data["weights"]]), data["params"]))
    history = model.fit(X_train, Y_train, epochs=2)
    return history

def main():
    print("start client")
    parser = argparse.ArgumentParser(description="parser to let client read its correct data")
    parser.add_argument('-d', '--data_id', type=int, help='index of data', required=True)
    args = parser.parse_args()
    print(f"data index = {args.data_id}")

    print("load data")
    loaded_data = read_csv_by_index("..\\dane\\generated_data", args.data_id)
    data, labels = create_dataset(loaded_data)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    batch_size = len(x_train)//10
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
                    history = function2(model, data, x_train, y_train)
                if data["header"]=="2":
                    history = function2(model, data, x_train, y_train)

                data_to_send = json.dumps({"weights": [w.tolist() for w in model.get_weights()],
                                           "summary": history.history})
                send_full_data(sock, data_to_send)
                print("Sent updated weights + history to server")

            if data["method"] == "fedprox":
                if data["header"] == "1":
                    model = function1(data)
                    history, error = function2betterprox(model, data, train_dataset)
                if data["header"] == "2":
                    history, error = function2betterprox(model, data, train_dataset)
                data_to_send = json.dumps({"weights": convert_np2list(model.get_weights()),
                                           "summary": history,
                                           "error": error})

                send_full_data(sock, data_to_send)
                print("Sent updated weights + history to server")
            if data["method"] == "fedpaq_float":
                if data["header"] == "1":
                    model = function1(data)
                    history = function2paqfloat(model, data, x_train, y_train)
                if data["header"] == "2":
                    history = function2paqfloat(model, data, x_train, y_train)

                data_to_send = json.dumps({"weights": [w.tolist() for w in model.simple_quantize_floats(model.get_weights())],
                                           "summary": history.history})
                send_full_data(sock, data_to_send)
                print("Sent updated weights + history to server")

            if data["method"] == "fedpaq_int":
                if data["header"] == "1":
                    model = function1(data)
                    history = function2paqint(model, data, x_train, y_train)
                if data["header"] == "2":
                    history = function2paqint(model, data, x_train, y_train)
                q_weights, params = model.quantize_weights_int(model.get_weights())
                data_to_send = json.dumps({"weights": [w.tolist() for w in q_weights],
                                           "params": params,
                                           "summary": history.history})
                send_full_data(sock, data_to_send)
                print("Sent updated weights + history to server")

if __name__ == "__main__":
    main()
