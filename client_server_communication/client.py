import socket
import json
import time


def train_local_model():
    # Simulated local model training (placeholder)
    return {"weights": [0.1, 0.2, 0.3]}


def main():
    while True:
        time.sleep(60)  # Simulating training time

        # Train local model
        local_model = train_local_model()
        print(f"Local model trained: {local_model}")

        # Send local model to the server
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('192.168.2.147', 8079))  # Private server ip address and it's port
        client_socket.send(json.dumps(local_model).encode())

        # Receive updated global model from server
        # 4096 is how much bytes up to it's able to receive
        global_model_data = client_socket.recv(4096).decode()
        global_model = json.loads(global_model_data)
        print(f"Received global model from server: {global_model}")

        client_socket.close()


if __name__ == "__main__":
    main()
