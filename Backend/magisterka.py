import sys

from flask import Flask, request
import flask
from flask_cors import CORS
import json
from tensorflow import keras
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic2.utilities import recreate_architecture_from_json


app = Flask(__name__)
CORS(app)

train = [[1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2],
         [3, 3, 3, 3, 3],
         [4, 4, 4, 4, 4]]

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

