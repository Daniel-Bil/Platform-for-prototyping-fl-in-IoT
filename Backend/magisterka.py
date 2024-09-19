import sys

from flask import Flask, request
import flask
from flask_cors import CORS
import json
import tensorflow as tf
import tensorflow.keras.layers as Layers
from tensorflow import keras
import os
from colorama import Fore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Backend.custom_model import CustomModel

app = Flask(__name__)
CORS(app)

train = [[1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2],
         [3, 3, 3, 3, 3],
         [4, 4, 4, 4, 4]]


def testing_generate_model(model, layer):
    model.add(layer)
    return model


def testing_train_model():
    raise Exception(NotImplementedError)


def testing_evaluate_model():
    raise Exception(NotImplementedError)


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
    model = CustomModel(inputs=input_layer, outputs=layer, name=name)
    return model


def recreate_architecture_from_json(path):
    with open(path, 'r') as file:
        data = json.load(file)[0]
    print(data)
    layers = []
    # print(Fore.BLUE,type(data),Fore.RESET)
    # print(data[0]["input_shape"])
    # print(tuple(data[0]["input_shape"]))
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


@app.route("/create", methods=['POST'])
def create():
    recreate_architecture_from_json(f"architectureJsons\\con_test.json")
    return flask.Response(response=json.dumps({"les": "go"}), status=201)

@app.route("/Architecture", methods=['GET', 'POST'])
def get_architecture():
    print("get architecture")

    if request.method == "POST":
        model = keras.Sequential()
        received_data = request.get_json()
        print(f"received data")
        # print(received_data["Architecture"])
        print()
        print(received_data)

        with open(f"architectureJsons\\{received_data['name']}.json", "w") as file:
            json.dump(received_data["architecture"], file, indent=4)
        from tensorflow.keras import layers, Model


        return_data = {
            "status": "success",
            "message": "success"
        }
        return flask.Response(response=json.dumps(return_data), status=201)

@app.route("/UpdateArchitecture", methods=["POST"])
def update_architecture():
    print("update_architecture")
    received_data = request.get_json()
    architecture_name = received_data['name']
    file_path = f"architectureJsons\\{architecture_name}.json"

    # Check if the architecture exists
    if os.path.exists(file_path):
        # Save the updated architecture to the file
        with open(f"architectureJsons\\{received_data['name']}.json", "w") as file:
            json.dump(received_data["architecture"], file, indent=4)

        return_data = {
            "status": "success",
            "message": f"Architecture '{architecture_name}' updated successfully."
        }
        return flask.Response(response=json.dumps(return_data), status=200)
    else:
        return_data = {
            "status": "error",
            "message": f"Architecture '{architecture_name}' does not exist."
        }
        return flask.Response(response=json.dumps(return_data), status=404)
    
@app.route("/DeleteArchitecture", methods=["DELETE"])
def delete_architecture():
    print("delete_architecture")
    received_data = request.get_json()
    print(f"received data")

    architecture_name = received_data['name']
    file_path = f"architectureJsons\\{architecture_name}.json"
    if os.path.exists(file_path):
        # Delete the file
        os.remove(file_path)

        return_data = {
            "status": "success",
            "message": f"Architecture '{architecture_name}' deleted successfully."
        }
        return flask.Response(response=json.dumps(return_data), status=200)
    else:
        return_data = {
            "status": "error",
            "message": f"Architecture '{architecture_name}' does not exist."
        }
        return flask.Response(response=json.dumps(return_data), status=404)
    #get request data
    #check if name exist
    #delete as this file
    #return info all good or not 

@app.route("/ArchitecturesNames", methods=['GET'])
def get_architecture_names():
    print("GET ArchitectureNames")
    architectures = os.listdir(f"{os.getcwd()}\\architectureJsons")
    print(architectures)
    architecturesList = []
    for architecture in architectures:
        with open(f"architectureJsons\\{architecture}", 'r') as file:
            data = json.load(file)
            architecturesList.append({"name": architecture.split(".")[0], "architecture_data": data})
    return flask.Response(response=json.dumps(architecturesList), status=201)


# # @app.route("/ArchitectureUpdate", methods=['POST'])
# # def update_architecture():
#     raise NotImplementedError


# @app.route("/ArchitectureDelete", methods=['POST'])
# def delete_architecture():
#     raise NotImplementedError


if __name__ == "__main__":
    app.run("localhost", 6969)

