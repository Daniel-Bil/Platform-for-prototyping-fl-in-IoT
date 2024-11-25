import asyncio

from colorama import Fore

async def START_WAITER(shared_state: dict) -> None:
    while shared_state['global_weights'] is None:
        print(f"{Fore.LIGHTMAGENTA_EX}SHORT W8 FOR GLOBAL WEIGHTS{Fore.RESET}")
        await asyncio.sleep(0.1)  # Small delay to avoid busy-waiting


async def AVERAGING_WAITER(shared_state: dict, client_id: str) -> None:
    while len(shared_state['completed_average_clients']) < shared_state['total_clients']:
        print(
            f"a waiting {client_id} avg:{shared_state['completed_average_clients']} com:{shared_state['completed_clients']} ")
        await asyncio.sleep(0.1)






