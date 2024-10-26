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
    if isinstance(data, str):
        data = data.encode()

    # Send the length of the data first (4 bytes)
    sock.sendall(struct.pack('!I', len(data)))

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


def main():
    print("start client")
    parser = argparse.ArgumentParser(description="parser to let client read its correct data")
    parser.add_argument('-d', '--data_id', type=int, help='index of data', required=True)
    parser.add_argument('--server_ip', type=str, default='localhost', help='Server IP address')
    parser.add_argument('--server_port', type=int, default=8100, help='Server port')
    args = parser.parse_args()

    print(f"Loading data for index {args.data_id}...")

    loaded_data = read_csv_by_index("..\\dane\\generated_data", args.data_id)
    data, labels = create_dataset(loaded_data)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(('localhost', args.server_port))  # Connect to the server
        model = None

        while True:

            # Receive weights from server
            print(f"{Fore.YELLOW}Receive data{Fore.RESET}")
            received_data = receive_full_data(sock)

            data = json.loads(received_data)
            print(f"{Fore.LIGHTCYAN_EX}Received data from server:{data['id']} {data['name']} hierfavg {data['header']}{Fore.RESET}")
            if data["header"] == "3":
                print(f"{Fore.LIGHTRED_EX} CLOSE CLIENT{Fore.RESET}")
                break


            if data["header"] == "1":
                model = function1(data)

            _ = function2(model, data, x_train, y_train)

            data_to_send = json.dumps({"source": "client",
                                       "weights": [w.tolist() for w in model.get_weights()]})

            send_full_data(sock, data_to_send)
            print("Sent updated weights + history to server")


if __name__ == "__main__":
    main()
