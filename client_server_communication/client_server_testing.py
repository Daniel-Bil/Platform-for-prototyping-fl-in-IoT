import os
import subprocess
import time
def open_terminal_and_run(script_name):
    subprocess.Popen(f'start cmd /k "title {script_name} && python {script_name}"', shell=True)

if __name__ == "__main__":
    # Start the server in a new terminal
    print(os.getcwd())
    print("Starting server...")
    # open_terminal_and_run("server.py")
    open_terminal_and_run("Aserver.py")

    # Give the server some time to start up
    time.sleep(3)

    # Start the two clients in separate terminals
    print("Starting client 1...")
    open_terminal_and_run("client.py")

    time.sleep(0.1)

    print("Starting client 2...")
    open_terminal_and_run("client.py")