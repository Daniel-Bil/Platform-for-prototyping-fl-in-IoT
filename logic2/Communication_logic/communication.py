import asyncio
import struct

from colorama import Fore


async def send_full_data(writer, data, client_id=""):
    print(f"{Fore.GREEN}send_full_data{Fore.RESET}")
    print("sFirst 40 bytes = ", data[:40])
    print("sLast 40 bytes = ", data[-40:])
    data = data.encode("utf-8")

    # Unpack the length of the message
    print(f"{Fore.LIGHTBLUE_EX}c_id[{client_id}] Expecting to send {len(data)} bytes{Fore.RESET}")

    writer.write(struct.pack('!I', len(data)))  # '!I' means big-endian unsigned int

    await writer.drain()

    writer.write(data)
    await writer.drain()


# Function to receive full data
async def receive_full_data(reader, client_id=''):
    print(f"{Fore.GREEN}receive_full_data{Fore.RESET}")
    try:
        # Read the length of the incoming data (4 bytes for the message length)
        raw_msglen = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        print("Connection closed before receiving the full message length.")
        return None  # Indicate that the connection was closed

    if not raw_msglen:
        print("Received no data for message length.")
        return None

    # Unpack the length of the message
    msglen = struct.unpack('!I', raw_msglen)[0]
    print(f"{Fore.LIGHTBLUE_EX}c_id[{client_id}] Expecting to receive {msglen} bytes{Fore.RESET}")

    try:
        # Read the exact message length
        data = await reader.readexactly(msglen)
    except asyncio.IncompleteReadError:
        print(f"{Fore.RED}Connection closed before receiving the full message.{Fore.RED}")
        return None

    # Print the first and last 30 bytes for debugging
    print("rFirst 40 bytes = ", data[:40])
    print("rLast 40 bytes = ", data[-40:])
    print(f"{Fore.LIGHTBLUE_EX}c_id[{client_id}] Received {len(data)} bytes{Fore.RESET}")
    return data.decode("utf-8")