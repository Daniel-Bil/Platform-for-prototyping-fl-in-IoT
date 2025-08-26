import subprocess
import time

def open_terminal_and_run(script_name, port=None, main_server_port=None, client_port=None, server_port=None, data_id=None, k_value=None):
    command = f'start cmd /k "title {script_name} && python {script_name}'
    if port is not None:
        command += f' --port {port}'
    if main_server_port is not None:
        command += f' --mainServer {main_server_port}'
    if client_port is not None:
        command += f' --client {client_port}'
    if server_port is not None:
        command += f' --server_port {server_port}'
    if data_id is not None:
        command += f' --data_id {data_id}'
    if k_value is not None:
        command += f' --k {k_value}'
    command += '"'
    subprocess.Popen(command, shell=True)

if __name__ == "__main__":
    # Start Main Server on port 8111
    print("Starting Main Server on port 8111...")
    open_terminal_and_run("Main_server_hierfavg.py", port=8111)
    time.sleep(1)  # Allow time for the main server to initialize

    # Start Edge Server 1 (connects to main server on 8111, listens to clients on 8100)
    print("Starting Edge Server 1 on client port 8100...")
    open_terminal_and_run("Edge_server_hierfavg.py", main_server_port=8111, client_port=8100)
    time.sleep(0.1)

    # # Start Edge Server 2 (connects to main server on 8111, listens to clients on 8101)
    print("Starting Edge Server 2 on client port 8101...")
    open_terminal_and_run("Edge_server_hierfavg.py", main_server_port=8111, client_port=8101)
    time.sleep(0.1)

    # Start Client for Edge Server 1 (connects to Edge Server 1 on port 8100)
    print("Starting Client 1 for Edge Server 1...")
    open_terminal_and_run("client_hierfavg.py", server_port=8100, data_id=1)
    time.sleep(0.1)

    # Start Client for Edge Server 2 (connects to Edge Server 2 on port 8101)
    print("Starting Client 2 for Edge Server 1...")
    open_terminal_and_run("client_hierfavg.py", server_port=8101, data_id=2)
