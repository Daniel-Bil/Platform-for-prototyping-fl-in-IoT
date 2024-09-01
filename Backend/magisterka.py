from flask import Flask, request
import flask
from flask_cors import CORS
import json
import tensorflow as tf
import tensorflow.keras.layers as Layers
from tensorflow import keras
import os
app = Flask(__name__)
CORS(app)


train = [[1,1,1,1,1],
         [2,2,2,2,2],
         [3,3,3,3,3],
         [4,4,4,4,4]]



def testing_generate_model(model, layer):
    model.add(layer)
    return model

def testing_train_model():
    raise Exception(NotImplementedError)

def testing_evaluate_model():
    raise Exception(NotImplementedError)

def testing_generate_layer(layerType, i=0):
    """
    Generate layer from given dictionary 
    """
    print(f"{layerType['name']}")
    print(type(layerType))
    kwargs = {}
    for key in list(layerType.keys()):
        if not (key == "id" or key == "name" or key=="kwargs"): 
            print(key, layerType[key]["default"])
            if (layerType[key]["type"]=="tuple"):
                kwargs[key]=tuple([int(l) for l in layerType[key]["default"]])
            elif(layerType[key]["type"]=="int"):
                kwargs[key]=int(layerType[key]["default"])
            elif(layerType[key]["type"]=="float"):
                kwargs[key]=float(layerType[key]["default"])
            else:
                kwargs[key]=layerType[key]["default"]
            
    layer = getattr(Layers, f"{layerType['name']}")
    if i==1:
        layer = layer(**kwargs, input_shape=(10,))
    else:
        layer = layer(**kwargs)
        
    return layer
    
    


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
            architecturesList.append({"name":architecture.split(".")[0], "architecture_data":data})
    return flask.Response(response=json.dumps(architecturesList), status=201)

if __name__ == "__main__":
    app.run("localhost", 6969)

   