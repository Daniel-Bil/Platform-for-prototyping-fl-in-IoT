# tools-deployment

Real LAN deployment implementation of the thesis federated-learning platform.
`tools2/` remains untouched and is the single-machine simulation/reference path.

## Algorithms

The cloud server selects the method at runtime:

- `FedAvg` — ordinary local training + sample-weighted FedAvg.
- `FedProx` — proximal term on each client; cloud aggregation remains sample-weighted FedAvg.
- `FedPAQ` — ordinary local training; updates are really quantized before LAN transport and dequantized at the cloud.
- `FedMA` — ordinary local training; cloud aligns hidden filters/neurons using Hungarian matching before averaging.
- `HierFedAvg` — real three-level topology: cloud -> edge aggregators -> training clients. Child models are weighted by local sample count at the edge, and edge models are weighted by the sum of child samples represented by that edge at the cloud.

There is no configured client count in the Python server. Every round snapshots the participants currently connected at that moment.

## Federated evaluation

After each successful aggregation (configurable with `--evaluate-every`) the cloud sends the new global model back to the successful participants for evaluation.

Raw test data and predictions never leave a client. Each client returns only:

- test sample count,
- binary cross-entropy loss,
- TP / TN / FP / FN,
- evaluation duration.

The cloud reconstructs exact aggregate accuracy, precision, recall, specificity, anomaly F1, normal-class F1 and macro F1 from those sufficient statistics.

For `HierFedAvg`, evaluation also follows cloud -> edge -> client and the edge aggregates its children's sufficient statistics before returning them to the cloud.

## Layout

```text
tools-deployment/
├── client/          # training + local evaluation client
├── edge/            # HierFedAvg edge aggregator
├── server/          # cloud server + aggregation algorithms + result writer
├── common/          # protocol, data, metrics, model, quantization, session registry
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

Tests cover binary transport, sample-weighted FedAvg, metric aggregation, FedPAQ quantization, FedMA permutation matching, dynamic late joins, distributed evaluation/result files, and sample-weighted HierFedAvg cloud->edge->children behavior.

## Direct topology: FedAvg / FedProx / FedPAQ / FedMA

Cloud:

```bash
python server/server.py \
  --host 0.0.0.0 \
  --port 8090 \
  --model config/default_model.json \
  --algorithm FedAvg \
  --rounds 3 \
  --local-epochs 1 \
  --seed 42 \
  --evaluate-every 1
```

Client:

```bash
python client/client.py \
  --server 192.168.2.231 \
  --port 8090 \
  --client-id device-1 \
  --data ../tools2/data/fl_dataset/client_df_RuralIoT_001
```

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

The edge forwards the global model, collects child updates concurrently, performs sample-weighted edge aggregation, and reports the exact represented sample count to the cloud. The cloud then performs sample-weighted aggregation across edge updates.

## Result artifacts

Every cloud start creates an isolated directory:

```text
results/<UTC timestamp>_<algorithm>/
├── config.json
├── rounds.csv
├── participants.csv
├── summary.json
└── global_weights.npz
```

### `config.json`

Contains experiment parameters, algorithm-specific parameters, random seed, model path, topology and the Git commit hash when available.

### `rounds.csv`

One row per global round, including:

- cohort and successful participant counts,
- represented train/test samples,
- global test loss,
- accuracy / precision / recall / specificity,
- anomaly F1 / normal F1 / macro F1,
- confusion matrix counts,
- training/evaluation/aggregation/round wall time,
- actual protocol bytes for training and evaluation,
- for HierFedAvg, both cloud<->edge and edge<->client traffic.

### `participants.csv`

One row per successful cloud participant per round with local training/evaluation measurements. For direct algorithms the participant is a client; for HierFedAvg it is an edge representing its child cohort.

### `summary.json`

Compact run summary containing the final global metrics and total communication/time counters across all completed rounds.

### `global_weights.npz`

Latest aggregated global model weights.

## Ansible experiment defaults

`ansible/group_vars/all.yml` exposes:

```yaml
fl_seed: 42
fl_evaluate_every: 1
fl_eval_batch_size: 256
fl_evaluation_timeout: 300
```

These can be overridden per run with `-e`, just like `fl_algorithm` and `fl_client_count`.

## FedProx diagnostic

FedProx uses the standard proximal local objective
`F_k(w) + (mu/2) * ||w - w_global||^2` and the cloud still performs
sample-weighted FedAvg aggregation. The deployment implementation runs the
custom proximal `train_step` through Keras `fit()` so the batch loop stays in
TensorFlow/Keras rather than crossing Python -> TensorFlow once per batch.

On a provisioned client, compare FedAvg, FedProx(mu=0) and the configured
FedProx value from identical initial weights/seeds:

```bash
cd ~/Platform-for-prototyping-fl-in-IoT
.venv-deployment/bin/python tools-deployment/scripts/check_fedprox.py \
  --data tools2/data/fl_dataset/client_df_RuralIoT_001 \
  --mu 0.01 --epochs 1 --batch-size 32
```

`FedProx(mu=0)` should be very close to ordinary local FedAvg training. The
script prints weight deltas, timings, base loss, the FedProx objective and the
raw proximal term.

## Run all deployment algorithms and collect results

From the operator laptop, after pushing/updating the remote repository:

```bash
python tools-deployment/scripts/run_all_experiments.py \
  --clients 7 \
  --rounds 5 \
  --local-epochs 3 \
  --repetitions 1
```

By default this runs, sequentially, `FedAvg`, `FedProx`, `FedPAQ`, `FedMA` and
`HierFedAvg` through the same Ansible launcher. Each experiment is allowed to
finish before the next begins. The newly-created result directory is copied
back from `main` and stored under:

```text
benchmark-results/<UTC timestamp>/
  manifest.json
  comparison.csv
  comparison.md
  aggregate.csv
  rep-001-seed-42/
    FedAvg/<server run directory>/...
    FedProx/<server run directory>/...
    FedPAQ/<server run directory>/...
    FedMA/<server run directory>/...
    HierFedAvg/<server run directory>/...
```

Use `--repetitions 3` (or more) for repeated measurements. All algorithms in a
single repetition receive the same seed; the seed increments between
repetitions. Important options include `--fedprox-mu`, `--fedpaq-bits`,
`--edge-count`, `--batch-size` and `--algorithms`.
