from flask import Flask, request
import flask
from flask_cors import CORS
import json
import tensorflow as tf
import tensorflow.keras.layers as Layers
from tensorflow import keras
import os
from colorama import Fore
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
    print(f"{layerType['name']}")
    print(type(layerType))
    kwargs = {}
    for key in list(layerType.keys()):
        if not (key == "id" or key == "name" or key == "kwargs" or key =="input_shape"):
            print(key, layerType[key]["default"])
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


def recreate_architecture_from_json(path):
    with open(path, 'r') as file:
        data = json.load(file)[0]
    print(data)
    layers = []
    # print(Fore.BLUE,type(data),Fore.RESET)
    # print(data[0]["input_shape"])
    # print(tuple(data[0]["input_shape"]))
    input_layer = tf.keras.layers.Input(shape=(60,))
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
            print(Fore.GREEN,"xDDDDDDDDDDDD",Fore.RESET)
            print(rest_of_data[z])
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
    return flask.Response(response=json.dumps({"les":"go"}), status=201)

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

        # input_shape = 10  # Replace with your input shape
        # input_layer = layers.Input(shape=(input_shape,))

        # # Define parent layer
        # parent_layer = layers.Dense(64, activation='relu')(input_layer)

        # # Left branch with 3 layers
        # left_branch = layers.Dense(32, activation='relu')(parent_layer)
        # left_branch = layers.Dense(32, activation='relu')(left_branch)
        # left_branch = layers.Dense(32, activation='relu')(left_branch)

        # # Right branch with 2 layers
        # right_branch = layers.Dense(32, activation='relu')(parent_layer)
        # right_branch = layers.Dense(32, activation='relu')(right_branch)

        # # Merge branches
        # merged = layers.Concatenate()([left_branch, right_branch])

        # # Final output layer
        # output_layer = layers.Dense(1, activation='sigmoid')(merged)

        # # Create the model
        # model = Model(inputs=input_layer, outputs=output_layer)

        # # Compile the model
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # # Print the model summary
        # model.summary()

        # testing_generate_layer(received_data[1])

        # for i in range(len(received_data)):
        #     print("i = ",i)
        #     if not i==0:
        #         model = testing_generate_model(model, testing_generate_layer(received_data[i],i))
        # model.summary()
        # model.compile(optimizer='adam',
        #       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #       metrics=[keras.metrics.SparseCategoricalAccuracy()])
        # model.summary()

        return_data = {
            "status": "success",
            "message": "success"
        }
        return flask.Response(response=json.dumps(return_data), status=201)


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


@app.route("/ArchitectureUpdate", methods=['POST'])
def update_architecture():
    raise NotImplementedError


@app.route("/ArchitectureDelete", methods=['POST'])
def delete_architecture():
    raise NotImplementedError


if __name__ == "__main__":
    app.run("localhost", 6969)

