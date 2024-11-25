import json

import tensorflow as tf
from pathlib import Path

from colorama import Fore

import tensorflow.keras.layers as Layers

def load_architectures(path: Path) -> list:
    print("load_architectures")
    architectures = []
    for idx, architecture in enumerate(path.glob("*.json")):
        if idx < 2:
            with open(architecture, 'r') as file:
                architectures.append({"name": path.stem, "data": json.load(file)[0]})

    return architectures


def load_methods():
    # return ["fedpaq_int", "fedpaq_float", "fedavg", "fedprox"]
    # return ["fedpaq_int", "fedavg", "fedpaq_float", "fedprox", "fedma"]
    return ["fedavg"]


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


def recreate_architecture_from_json2(data, name=""):
    print(f"{Fore.BLUE}recreate_architecture_from_json2 {name}{Fore.RESET}")
    layers = []

    input_layer = tf.keras.layers.Input(shape = eval(data[0]["input_shape"]))
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