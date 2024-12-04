import json

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from colorama import Fore

import tensorflow.keras.layers as Layers
from pandas import DataFrame


def load_architectures(path: Path) -> list:
    print("load_architectures")
    architectures = []
    for idx, architecture in enumerate(path.glob("*.json")):
        if idx < 2:
            with open(architecture, 'r') as file:
                architectures.append({"name": architecture.stem, "data": json.load(file)[0]})

    return architectures


def load_methods():
    # return ["fedpaq_int", "fedpaq_float", "fedavg", "fedprox"]
    return ["fedavg", "fedpaq_int", "fedpaq_float", "fedprox", "fedma"]
    # return ["fedavg"]


def write_to_tensorboard(history_data: dict, log_dir: Path, start_step: int = 0):
    # Create a TensorBoard summary writer
    writer = tf.summary.create_file_writer(str(log_dir))

    # Use the summary writer to log metrics, incrementing the step for each iteration
    with writer.as_default():
        for step, (acc, loss) in enumerate(zip(history_data['accuracy'], history_data['loss']), start=start_step):
            tf.summary.scalar('accuracy', acc, step=step)
            tf.summary.scalar('loss', loss, step=step)
            writer.flush()


def testing_generate_layer(layerType):
    """
    Generate layer from given dictionary
    """
    print(f"{Fore.LIGHTGREEN_EX}{layerType['name']}{Fore.RESET}")
    kwargs = {}
    for key in list(layerType.keys()):
        if not (key == "id" or key == "name" or key == "kwargs" or key =="input_shape"):
            if (layerType[key]["type"] == "tuple"):
                kwargs[key] = tuple([int(l) for l in layerType[key]["default"]])

            elif (layerType[key]["type"] == "int"):
                kwargs[key] = int(layerType[key]["default"])

            elif (layerType[key]["type"] == "float"):
                kwargs[key] = float(layerType[key]["default"])

            else:
                kwargs[key] = layerType[key]["default"]

    layer = getattr(Layers, f"{layerType['name']}")
    layer = layer(**kwargs)

    return layer


def recreate_architecture_from_json2(data: dict, name: str = ""):
    print(f"{Fore.BLUE}recreate_architecture_from_json2 {name}{Fore.RESET}")
    layers = []

    input_layer = tf.keras.layers.Input(shape=eval(data[0]["input_shape"]))
    layer = testing_generate_layer(data[0])(input_layer)
    for i in range(len(data)):
        if i == 0:
            continue
        if not isinstance(data[i], list):
            if not data[i]["name"] == "Concatenate":
                layer = testing_generate_layer(data[i])(layer)
            else:
                layer = tf.keras.layers.Concatenate()(layers)
                layers = []

        if isinstance(data[i], list):
            for j in range(len(data[i])):
                layers.append(recreate_branch(layer, data[i][j]))
    model = tf.keras.Model(inputs=input_layer, outputs=layer, name=name)
    return model


def recreate_architecture_from_json(path: str):
    with open(path, 'r') as file:
        data = json.load(file)[0]
    print(data)
    layers = []
    input_layer = tf.keras.layers.Input(shape = eval(data[0]["input_shape"]))
    print(input_layer)
    layer = testing_generate_layer(data[0])(input_layer)
    print(layer)
    for i in range(len(data)):
        if i == 0:
            continue
        print(data[i])
        if not isinstance(data[i], list):
            if not data[i]["name"] == "Concatenate":
                layer = testing_generate_layer(data[i])(layer)
            else:
                layer = tf.keras.layers.Concatenate()(layers)
                layers = []

        if isinstance(data[i], list):
            for j in range(len(data[i])):
                layers.append(recreate_branch(layer, data[i][j]))
    model = tf.keras.Model(inputs=input_layer, outputs=layer)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()

def recreate_branch(parent_layer, rest_of_data):
    if len(rest_of_data) == 0:
        return parent_layer
    layers = []
    layer = parent_layer
    for z in range(len(rest_of_data)):
        if not isinstance(rest_of_data[z], list):
            if not rest_of_data[z]["name"] == "Concatenate":
                layer = testing_generate_layer(rest_of_data[z])(layer)
            else:
                layer = tf.keras.layers.Concatenate()(layers)
                layers = []

        if isinstance(rest_of_data[z], list):
            for j in range(len(rest_of_data[z])):
                layers.append(recreate_branch(layer, rest_of_data[z][j]))
    return layer


def read_csv_by_index(path: Path, index: int) -> DataFrame:
    # Get a list of all CSV files in the directory

    paths = [csv_file for csv_file in path.glob("*.csv")]

    # Check if the provided index is within the range
    if index < 0 or index >= len(paths):
        raise IndexError("Index out of range. Please provide a valid index.")

    # Get the filename at the specified index
    selected_file = paths[index]

    # Read the CSV file
    df = pd.read_csv(selected_file)
    print(f"Loaded file: {selected_file}")

    return df


def create_dataset(df: DataFrame, window_size: int = 5) -> tuple[np.ndarray,np.ndarray]:
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
