import argparse
import asyncio
import json
import os
import sys
import numpy as np
from colorama import Fore
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic2.weights_operations import weights2list, list2np
from logic2.Communication_logic.communication import send_full_data, receive_full_data

lock = asyncio.Lock()


async def start_lock(shared_state):
    while shared_state['start'] is None and len(shared_state['completed_iterations_clients']) > 0:
        print(f"{Fore.LIGHTMAGENTA_EX}waiting for start or remove of completed iterations{Fore.RESET}")
        await asyncio.sleep(0.1)


async def shared_lock_2(shared_state):
    while len(shared_state['completed_architecture']) < shared_state['total_clients']:
        print(f"{Fore.LIGHTMAGENTA_EX}waiting to go to next arch{Fore.RESET}")
        await asyncio.sleep(0.5)


async def wait_for_end_of_iterations(shared_state):
    while len(shared_state['completed_iterations_clients']) < shared_state['total_clients']:
        print(f"{Fore.LIGHTMAGENTA_EX}Iterations are not finished{Fore.RESET}")
        await asyncio.sleep(0.2)


async def wait_for_all_clients_receive_data(shared_state, client_id="placeholde"):
    while len(shared_state['completed_receiving_clients']) < shared_state['total_clients']:
        print(f"c waiting {client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_receiving_clients']}")
        await asyncio.sleep(0.1)  # Small delay to avoid busy-waiting
    else:
        print(f"{Fore.LIGHTBLUE_EX} all clients received data : completed clients {shared_state['completed_receiving_clients']} {Fore.RESET}")


async def wait_for_all_clients_aggregate_weights(shared_state, client_id="placeholde"):
    while len(shared_state['completed_average_clients']) < shared_state['total_clients']:
        print(f"a waiting {client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_receiving_clients']} ")
        await asyncio.sleep(0.1)
    print(f"{Fore.LIGHTBLUE_EX} all clients aggregated weights : completed clients {shared_state['completed_average_clients']} {Fore.RESET}")


async def wait_for_completed_removal_receive(shared_state, client_id="placeholde"):
    while len(shared_state['completed_receiving_clients']) > 0:
        print(f"waiting {client_id} for others to remove itself com:{shared_state['completed_receiving_clients']}")
        await asyncio.sleep(0.1)
    print(f"all removed themselves from {Fore.LIGHTGREEN_EX}completed{Fore.RESET}")

async def wait_for_completed_removal_aggregation(shared_state, client_id="placeholde"):
    while len(shared_state['completed_average_clients']) > 0:
        print(f"waiting {client_id} for others to remove itself com:{shared_state['completed_average_clients']}")
        await asyncio.sleep(0.1)
    print(f"all removed themselves from {Fore.LIGHTGREEN_EX}aggregated{Fore.RESET}")


async def wait_for_completed_removal_iterations(shared_state, client_id="placeholde"):
    while len(shared_state['completed_iterations_clients']) > 0:
        print(f"waiting {client_id} for others to remove itself com:{shared_state['completed_iterations_clients']}")
        await asyncio.sleep(0.1)
    print(f"all removed themselves from {Fore.LIGHTGREEN_EX}iterations{Fore.RESET}")

async def handle_client(reader, writer, shared_state, K):
    client_info = writer.get_extra_info('peername')
    client_id = client_info[1]
    shared_state['client_ids'].append(client_id)
    shared_state['client_writers'].append(writer)  # Store client writer for broadcasting
    starting_main_server_data_received = "no"
    starting_main_server_data_send = "yes"
    while True:
        # await start_lock(shared_state) # lock1


        while starting_main_server_data_received == shared_state["main_server_data_received"] and not shared_state["finished"]:
            print(f"{Fore.LIGHTMAGENTA_EX}still same data from main server{Fore.RESET}")
            await asyncio.sleep(0.5)

        starting_main_server_data_received = shared_state["main_server_data_received"]



        if shared_state['finished']:
            print(f"{Fore.LIGHTCYAN_EX}END program{Fore.RESET}")
            await send_full_data(writer, json.dumps({"header": "3",
                                                     "name": shared_state["current_model_name"],
                                                     "method": "finished",
                                                     "id": client_id}))
            break

        for idx in range(K):

            if shared_state["main_server_data"]["header"] == "1":
                data_to_send = json.dumps({
                    "header": "1",
                    "name": shared_state["main_server_data"]["name"],
                    "data": shared_state["main_server_data"]['data'],
                    "weights": shared_state["main_server_data"]['weights'],
                    "number of client aggregations": 5,
                    "id": client_id
                })
            else:
                data_to_send = json.dumps({
                    "header": "2",
                    "name": shared_state["main_server_data"]["name"],
                    "weights": weights2list(shared_state['local_aggregated_weights']),
                    "number of client aggregations": 5,
                    "id": client_id
                })

            await send_full_data(writer, data_to_send)

            received_data = await receive_full_data(reader)
            received_data_json = json.loads(received_data)
            print(f"{Fore.LIGHTGREEN_EX}Received updated weights and history from client {client_id}{Fore.RESET}")

            # 1. receive
            # 2. add to clients
            # 3. check if all received
            # 4. aggregate if all received single client
            # 5. empty

            shared_state['weights'][client_id] = list2np(received_data_json["weights"])
            async with lock:
                print(f"{Fore.LIGHTBLUE_EX}append completed_receiving_clients{Fore.RESET}")
                shared_state['completed_receiving_clients'].append(client_id)

            await wait_for_all_clients_receive_data(shared_state, client_id)

            # weights aggregation performed by client 0
            if client_id == shared_state["client_ids"][0]:
                print(f"start averaging {client_id}")
                weights = []
                for client_id2 in shared_state['completed_receiving_clients']:
                    weights.append(list2np(shared_state['weights'][client_id2]))
                shared_state['local_aggregated_weights'] = np.mean(weights, axis=0)
                print("averaging done")
                print("test1111")
                if shared_state["main_server_data_send"] == "no":
                    shared_state["main_server_data_send"] = "yes"
                else:
                    shared_state["main_server_data_send"] = "no"

            async with lock:
                print(f"{Fore.LIGHTBLUE_EX}append average{Fore.RESET}")
                shared_state['completed_average_clients'].append(client_id)

            while len(shared_state['completed_average_clients']) < shared_state['total_clients']:
                print(
                    f"a waiting {client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_receiving_clients']} ")
                await asyncio.sleep(0.1)

            async with lock:
                shared_state['completed_receiving_clients'].remove(client_id)
                print(f"{Fore.LIGHTWHITE_EX} Removed {client_id} -> {shared_state['completed_receiving_clients']} completed_receiving_clients")

            await wait_for_completed_removal_receive(shared_state, client_id)

            async with lock:
                shared_state['completed_average_clients'].remove(client_id)
                print(f"{Fore.LIGHTWHITE_EX} Removed {client_id} -> {shared_state['completed_average_clients']} completed average")

            await wait_for_completed_removal_aggregation(shared_state, client_id)

            print(f"end iteration {idx+1} out of {K} for {client_id}")
            print(f"{client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_receiving_clients']}")

        print(f"finished all {K} iterations")

        async with lock:
            print(f"{Fore.LIGHTBLUE_EX}append completed iterations{Fore.RESET}")
            shared_state['completed_iterations_clients'].append(client_id)


        while starting_main_server_data_send == shared_state["main_server_data_send"]:
            print(f"{Fore.LIGHTMAGENTA_EX}still not sent data to main server{Fore.RESET}")
            await asyncio.sleep(0.1)


        async with lock:
            shared_state['completed_iterations_clients'].remove(client_id)
            print(f"{Fore.LIGHTWHITE_EX} Removed {client_id} -> {shared_state['completed_iterations_clients']} completed iterations")

        await wait_for_completed_removal_iterations(shared_state, client_id)



async def connect_to_main_server(main_server_port, shared_state):
    reader, writer = await asyncio.open_connection('localhost', main_server_port)
    print(f"Edge Server connected to Main Server on port {main_server_port}")

    # Keep receiving data from the main server and distributing it to clients
    while True:
        data = await receive_full_data(reader)
        data_json = json.loads(data)
        print(f"{Fore.LIGHTCYAN_EX}RECEIVE FROM MAIN SERVER{Fore.RESET}")
        shared_state["main_server_data"] = data_json
        shared_state["finished"] = data_json["finished"]

        if shared_state["main_server_data_received"] == "no":
            shared_state["main_server_data_received"] = "yes"
        else:
            shared_state["main_server_data_received"] = "no"

        print(f"{Fore.LIGHTCYAN_EX}RECEIVED FROM MAIN SERVER{Fore.RESET}")

        await wait_for_end_of_iterations(shared_state)

        print(f"{Fore.LIGHTCYAN_EX}SENDING TO MAIN SERVER{Fore.RESET}")
        data_to_send = {
            "weights": weights2list(shared_state["local_aggregated_weights"]),  # or aggregated weights
            "status": "completed"  # example status flag
        }

        await send_full_data(writer, json.dumps(data_to_send))

        if shared_state["main_server_data_send"] == "no":
            shared_state["main_server_data_send"] = "yes"
        else:
            shared_state["main_server_data_send"] = "no"



async def main(K):
    parser = argparse.ArgumentParser(description="Edge server for federated learning.")
    parser.add_argument("--mainServer", type=int, default=8111, help="Port for the main server to connect to")
    parser.add_argument("--client", type=int, default=8100, help="Port for clients to connect to this Edge Server")
    args = parser.parse_args()

    shared_state = {
        'weights': {},  # Store client weights
        'client_writers': [],  # Store client connections
        'client_ids': [],
        'completed_receiving_clients': [],
        'completed_iterations_clients': [],
        "completed_average_clients": [],
        'local_aggregated_weights': None,
        'global_aggregated_weights': None,
        'total_clients': 1,  # Number of clients this edge server handles
        "finished": False,
        "Main1": False,
        "Main2": True,
        "Main3": True,
        "main_server_data_received": "no",
        "main_server_data_send": "no",
        "Finished_iterations": False,
        "main_server_data": None
    }

    # Start listening for client connections
    server_to_clients = await asyncio.start_server(lambda r, w: handle_client(r, w, shared_state, K), 'localhost', args.client)

    # Start the task to connect to the main server
    asyncio.create_task(connect_to_main_server(args.mainServer, shared_state))

    async with server_to_clients:
        await server_to_clients.serve_forever()

if __name__ == "__main__":
    K = 5  # Number of local aggregation rounds before sending to the main server
    asyncio.run(main(K))
