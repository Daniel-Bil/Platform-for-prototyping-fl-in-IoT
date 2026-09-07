# tools-deployment

Distributed deployment implementation for the thesis FL platform.

`tools2/` remains untouched and continues to serve as the single-machine simulation/reference implementation. This directory implements the real LAN deployment path.

## Current scope

- Real TCP client/server communication.
- Persistent client connections.
- Dynamic number of clients: there is **no configured expected client count**.
- Clients joining during a round automatically wait for the next round.
- Binary compressed NumPy model transport (no weight arrays encoded as JSON).
- Sample-weighted FedAvg aggregation.
- Per-round timeout and disconnect handling.
- Per-run results directory with configuration, round metrics and latest global weights.
- Compatible with the model JSON format and FL dataset format already used by `tools2`. A copy of the current default model is kept in `config/default_model.json`; `tools2/` itself is not modified.

Current algorithm: **FedAvg only**. Other FL methods should be added after the distributed FedAvg path is verified on real machines.

The server is the source of truth for the model architecture, sequence length and feature list. A client first connects, receives that experiment configuration, prepares its own local dataset, then registers as ready. This prevents different machines from silently training incompatible models.

## Environment

Recommended: Python 3.9.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3.9 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Code smoke tests

The transport/aggregation/dynamic-client behavior can be checked without TensorFlow:

```bash
python tests/smoke_test.py
```

## First smoke test: server + one client

Assume the repository layout is unchanged and the big machine has LAN IP `192.168.1.50`.

### Big machine / server

From `tools-deployment/`:

```bash
python server/server.py \
  --host 0.0.0.0 \
  --port 8090 \
  --model config/default_model.json \
  --rounds 3 \
  --local-epochs 1
```

Open TCP port 8090 in the server firewall for the private LAN if necessary.

### Small machine / client

Copy the repository (or at minimum `tools-deployment/` plus one prepared client dataset) to the client machine, then run from `tools-deployment/`:

```bash
python client/client.py \
  --server 192.168.1.50 \
  --port 8090 \
  --client-id sensor-003 \
  --data ../tools2/data/fl_dataset/client_df_RuralIoT_003
```

The server waits until at least one client exists. Before every round it opens a short `--join-window` (default 2 seconds) and then snapshots all clients that are connected at that moment.

## Adding more clients

Start the exact same client program on any number of machines, using a unique `--client-id` and that machine's local dataset. The server command does not change.

Example second client:

```bash
python client/client.py \
  --server 192.168.1.50 \
  --client-id sensor-021 \
  --data ../tools2/data/fl_dataset/client_df_RuralIoT_21
```

If it connects while a round is already training, it automatically joins the next round.

## Useful server options

- `--rounds 0`: run indefinitely until Ctrl+C.
- `--join-window 5`: allow a longer period for newly started clients to join before each round snapshot.
- `--round-timeout 300`: maximum seconds to wait for a client update in one round.
- `--auth-token ...`: optional shared token. Prefer environment variable `FL_AUTH_TOKEN` instead of command history.

## Output

Each server start creates:

```text
results/<UTC timestamp>/
├── config.json
├── rounds.csv
└── global_weights.npz
```

`rounds.csv` records the cohort, successful updates, aggregation time and actual wire bytes transferred by the deployment protocol.

## Deployment semantics

```text
server starts
    |
    +-- waits for >= 1 registered client
    |
    +-- join window
    |
    +-- snapshot currently connected clients  <--- round cohort
    |
    +-- broadcast current global weights
    |
    +-- clients train concurrently
    |
    +-- collect valid updates until all respond or timeout
    |
    +-- sample-weighted FedAvg
    |
    +-- save round result and global weights
    |
    `-- next round
```

A new client never changes a cohort that is already training. A disconnected/timed-out client is skipped for that round; successful clients can still be aggregated.
