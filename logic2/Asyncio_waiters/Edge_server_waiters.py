import asyncio

from colorama import Fore


async def start_lock(shared_state: dict) -> None:
    while shared_state['start'] is None and len(shared_state['completed_iterations_clients']) > 0:
        print(f"{Fore.LIGHTMAGENTA_EX}waiting for start or remove of completed iterations{Fore.RESET}")
        await asyncio.sleep(0.1)


async def shared_lock_2(shared_state: dict) -> None:
    while len(shared_state['completed_architecture']) < shared_state['total_clients']:
        print(f"{Fore.LIGHTMAGENTA_EX}waiting to go to next arch{Fore.RESET}")
        await asyncio.sleep(0.5)


async def wait_for_end_of_iterations(shared_state: dict) -> None:
    while len(shared_state['completed_iterations_clients']) < shared_state['total_clients']:
        print(f"{Fore.LIGHTMAGENTA_EX}Iterations are not finished{Fore.RESET}")
        await asyncio.sleep(0.2)


async def wait_for_all_clients_receive_data(shared_state: dict, client_id: str = "placeholder") -> None:
    while len(shared_state['completed_receiving_clients']) < shared_state['total_clients']:
        print(f"c waiting {client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_receiving_clients']}")
        await asyncio.sleep(0.1)  # Small delay to avoid busy-waiting

    print(f"{Fore.LIGHTBLUE_EX} all clients received data : completed clients {shared_state['completed_receiving_clients']} {Fore.RESET}")


async def wait_for_all_clients_aggregate_weights(shared_state: dict, client_id: str = "placeholder") -> None:
    while len(shared_state['completed_average_clients']) < shared_state['total_clients']:
        print(f"a waiting {client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_receiving_clients']} ")
        await asyncio.sleep(0.1)
    print(f"{Fore.LIGHTBLUE_EX} all clients aggregated weights : completed clients {shared_state['completed_average_clients']} {Fore.RESET}")


async def wait_for_completed_removal_receive(shared_state: dict, client_id: str = "placeholder") -> None:
    while len(shared_state['completed_receiving_clients']) > 0:
        print(f"waiting {client_id} for others to remove itself com:{shared_state['completed_receiving_clients']}")
        await asyncio.sleep(0.1)
    print(f"all removed themselves from {Fore.LIGHTGREEN_EX}completed{Fore.RESET}")

async def wait_for_completed_removal_aggregation(shared_state: dict, client_id: str = "placeholder") -> None:
    while len(shared_state['completed_average_clients']) > 0:
        print(f"waiting {client_id} for others to remove itself com:{shared_state['completed_average_clients']}")
        await asyncio.sleep(0.1)
    print(f"all removed themselves from {Fore.LIGHTGREEN_EX}aggregated{Fore.RESET}")


async def wait_for_completed_removal_iterations(shared_state: dict, client_id: str = "placeholder") -> None:
    while len(shared_state['completed_iterations_clients']) > 0:
        print(f"waiting {client_id} for others to remove itself com:{shared_state['completed_iterations_clients']}")
        await asyncio.sleep(0.1)
    print(f"all removed themselves from {Fore.LIGHTGREEN_EX}iterations{Fore.RESET}")