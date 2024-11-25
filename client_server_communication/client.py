import os
import socket
import json
import sys

import struct
import pandas as pd
import tensorflow as tf
import argparse

import numpy as np
from colorama import Fore
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic2.Federated_logic.federated_methods_client import simple_training, fedProx_training, fedPaq_int_training, \
    fedPaq_float_training
from logic2.utilities import recreate_architecture_from_json2
from logic2.weights_operations import weights2list, quantize_weights_int, simple_quantize_floats



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


def send_full_data(writer, data):
    print(f"{Fore.GREEN}send_full_data{Fore.RESET}")
    print("sFirst 40 bytes = ", data[:40])
    print("sLast 40 bytes = ", data[-40:])
    data = data.encode()

    # Write the length of the data (4 bytes for the message length)
    writer.sendall(struct.pack('!I', len(data)))  # '!I' means big-endian unsigned int

    # Write the actual data
    writer.sendall(data)

# Function to receive full data
def receive_full_data(reader):
    print(f"{Fore.GREEN}receive_full_data{Fore.RESET}")
    try:
        # Read the length of the incoming data (4 bytes for the message length)
        raw_msglen = reader.recv(4)
        if not raw_msglen:
            print("Received no data for message length.")
            return None
    except Exception as e:
        print(f"{Fore.RED}Error reading message length: {e}{Fore.RESET}")
        return None

    # Unpack the length of the message
    msglen = struct.unpack('!I', raw_msglen)[0]
    print(f"{Fore.LIGHTBLUE_EX}Expecting to receive {msglen} bytes{Fore.RESET}")

    try:
        # Read the exact message length
        data = b""
        while len(data) < msglen:
            part = reader.recv(msglen - len(data))
            if not part:
                print(f"{Fore.RED}Connection closed before receiving the full message.{Fore.RESET}")
                return None
            data += part
    except Exception as e:
        print(f"{Fore.RED}Error reading message: {e}{Fore.RESET}")
        return None

    # Print the first and last 40 bytes for debugging
    print("rFirst 40 bytes = ", data[:40])
    print("rLast 40 bytes = ", data[-40:])

    return data.decode()


def build_model(data):
    print("function1")
    model = recreate_architecture_from_json2(data["data"]["data"], data["data"]["name"])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                        tf.keras.metrics.Precision(name='precision'),
                                                        tf.keras.metrics.Recall(name='recall')])
    return model


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

            if data["header"] == "1":
                model = build_model(data)

            if (data["method"] == "fedavg") or (data["method"] == "fedma"):
                history = simple_training(model, data, x_train, y_train)

                data_to_send = json.dumps({"weights": weights2list(model.get_weights()),
                                           "summary": history.history})


            if data["method"] == "fedprox":
                history, error = fedProx_training(model, data, train_dataset)
                data_to_send = json.dumps({"weights": weights2list(model.get_weights()),
                                           "summary": history,
                                           "error": error})


            if data["method"] == "fedpaq_float":
                history = fedPaq_float_training(model, data, x_train, y_train)

                data_to_send = json.dumps({"weights": weights2list(simple_quantize_floats(model.get_weights())),
                                           "summary": history.history})


            if data["method"] == "fedpaq_int":
                history = fedPaq_int_training(model, data, x_train, y_train)

                q_weights, params = quantize_weights_int(model.get_weights())

                data_to_send = json.dumps({"weights": weights2list(q_weights),
                                           "params": params,
                                           "summary": history.history})


            send_full_data(sock, data_to_send)
            print("Sent updated weights + history to server")


if __name__ == "__main__":
    main()
