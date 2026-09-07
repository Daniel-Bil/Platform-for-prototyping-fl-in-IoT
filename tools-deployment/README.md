# tools-deployment

Real LAN deployment implementation of the thesis federated-learning platform.
`tools2/` remains untouched and is the single-machine simulation/reference path.

## Algorithms

The cloud server has one implementation and selects the method at runtime:

- `FedAvg` — ordinary local training + sample-weighted FedAvg.
- `FedProx` — proximal term on each client; cloud aggregation remains sample-weighted FedAvg.
- `FedPAQ` — ordinary local training, but client updates are actually quantized to 8-bit integers (configurable 1-8 bits) before network transport; the cloud dequantizes and performs sample-weighted averaging.
- `FedMA` — ordinary local training; cloud aligns hidden filters/neurons using Hungarian matching before averaging, following `tools2/method_fedma.py`.
- `HierFedAvg` — real three-level topology: cloud -> edge aggregators -> training clients. Edge and cloud averages follow the `tools2/method_hierfavg.py` two-level semantics.

There is no configured client count in the Python server. Every round snapshots the participants currently connected at that moment.

## Layout

```text
tools-deployment/
├── client/          # training client
├── edge/            # HierFedAvg edge aggregator
├── server/          # cloud server + aggregation algorithms
├── common/          # protocol, data, model, quantization, session registry
├── config/
├── tests/
└── results/
```

## Environment

The Ansible deployment uses Python 3.10 because TensorFlow 2.14.1 does not provide Python 3.12 wheels.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## TensorFlow-free smoke tests

```bash
python tests/smoke_test.py
```

Tests cover binary transport, weighted FedAvg, FedPAQ quantization, FedMA permutation matching, dynamic late joins, and a full fake HierFedAvg cloud->edge->children round.

## Direct topology: FedAvg / FedProx / FedPAQ / FedMA

Cloud:

```bash
python server/server.py \
  --host 0.0.0.0 \
  --port 8090 \
  --model config/default_model.json \
  --algorithm FedAvg \
  --rounds 3 \
  --local-epochs 1
```

Client:

```bash
python client/client.py \
  --server 192.168.2.231 \
  --port 8090 \
  --client-id device-1 \
  --data ../tools2/data/fl_dataset/client_df_RuralIoT_001
```

Change only `--algorithm` on the cloud to run another direct method.

FedProx parameter:

```bash
--algorithm FedProx --fedprox-mu 0.01
```

FedPAQ parameter:

```bash
--algorithm FedPAQ --fedpaq-bits 8
```

## HierFedAvg topology

```text
main/cloud:8090
    |
    +---- edge-1:8091 ---- device-1, device-3, ...
    |
    `---- edge-2:8091 ---- device-2, device-4, ...
```

Cloud:

```bash
python server/server.py --host 0.0.0.0 --port 8090 \
  --model config/default_model.json --algorithm HierFedAvg --rounds 3
```

Edge:

```bash
python edge/edge.py \
  --edge-id edge-1 \
  --cloud 192.168.2.231 \
  --cloud-port 8090 \
  --listen-host 0.0.0.0 \
  --listen-port 8091
```

Normal `client/client.py` processes then connect to the edge's address/port. The edge forwards the global model, collects its children concurrently, aggregates their models locally, and sends one edge model to the cloud.

## Results

Every cloud start creates:

```text
results/<UTC timestamp>/
├── config.json
├── rounds.csv
└── global_weights.npz
```

`rounds.csv` records the selected algorithm, cohort, successful updates, aggregation time, and actual protocol bytes in both directions. FedPAQ therefore reports real reduced uplink traffic rather than a theoretical estimate.
