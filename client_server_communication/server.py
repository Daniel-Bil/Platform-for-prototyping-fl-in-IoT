import socket
import json


def update_global_model(local_models):
    # Placeholder function to aggregate local models and update global model
    global_model = {"weights": [sum(model["weights"]) / len(local_models) for model in local_models]}
    return global_model


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('192.168.2.147', 8079)) # Private server ip address and it's port
    server_socket.listen(5)
    print("Server is waiting for clients...")

    local_models = []

    while True:
        client_socket, address = server_socket.accept()
        print(f"Connection from {address} established.")

        # type in here how much bytes you want to accept from one client at a time
        data = client_socket.recv(4096).decode()
        local_model = json.loads(data)
        print(f"Received model from client: {local_model}")
        local_models.append(local_model)

        # After receiving updates from enough clients, update the global model
        if len(local_models) == 1:  # For example, wait for 1 clients
            global_model = update_global_model(local_models)
            print(f"Global model updated: {global_model}")

            # Send global model back to the client
            client_socket.send(json.dumps(global_model).encode())
            local_models = []  # Reset for next round of updates

        client_socket.close()


if __name__ == "__main__":
    main()
