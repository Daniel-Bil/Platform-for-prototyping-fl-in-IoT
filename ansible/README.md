# Ansible deployment

Inventory names match aliases from `~/.ssh/config`, so Ansible reuses your SSH
users/keys and the existing `ProxyJump main` setup. Git cloning uses the SSH
repository URL with forwarded local ssh-agent; private keys are not copied to
remote machines.

## 1. Provision machines

```bash
ansible-playbook \
  -i ansible/inventory.yml \
  ansible/playbooks/01-provision-repo.yml \
  --limit 'main:device-1'
```

Omit `--limit` to provision every machine. Provisioning checks out the
`deployment` branch and creates a Python 3.10 `.venv-deployment` for TensorFlow
2.14.1.

## 2. Update code after a push

```bash
ansible-playbook \
  -i ansible/inventory.yml \
  ansible/playbooks/03-update-repo.yml
```

The updater fetches and fast-forwards the deployment branch on `main` and all
`device-*` hosts, then synchronizes the existing Python environment. It refuses
to overwrite tracked source modifications made directly on a remote host.

## 3. Run an experiment

The same launcher runs every method and any supported number of dataset-backed
clients.

```bash
ansible-playbook \
  -i ansible/inventory.yml \
  ansible/playbooks/02-start-one-client.yml \
  -e fl_client_count=4 \
  -e fl_algorithm=FedAvg
```

Direct methods:

```bash
-e fl_algorithm=FedAvg
-e fl_algorithm=FedProx -e fl_fedprox_mu=0.01
-e fl_algorithm=FedPAQ -e fl_fedpaq_bits=8
-e fl_algorithm=FedMA
```

HierFedAvg:

```bash
ansible-playbook \
  -i ansible/inventory.yml \
  ansible/playbooks/02-start-one-client.yml \
  -e fl_client_count=7 \
  -e fl_algorithm=HierFedAvg \
  -e fl_edge_count=2
```

For seven clients and two edges:

```text
main/cloud
  +-- edge-1 on device-1 <- device-1, device-3, device-5, device-7
  `-- edge-2 on device-2 <- device-2, device-4, device-6
```

Before every experiment the launcher stops stale client and edge services on all
`device-*` hosts, restarts the cloud server, then starts only the selected
participants. `device-8` remains excluded until `fl_dataset_name` is assigned.

## Reproducible evaluation defaults

Defaults in `ansible/group_vars/all.yml`:

```yaml
fl_seed: 42
fl_evaluate_every: 1
fl_eval_batch_size: 256
fl_evaluation_timeout: 300
```

Override them per run with `-e`, for example:

```bash
-e fl_rounds=20 -e fl_local_epochs=3 -e fl_seed=123
```

## Result files

The cloud writes each run under:

```text
~/Platform-for-prototyping-fl-in-IoT/tools-deployment/results/<timestamp>_<algorithm>/
```

with `config.json`, `rounds.csv`, `participants.csv`, `summary.json` and
`global_weights.npz`.

## Dataset used by deployment experiments

`group_vars/all.yml` points `fl_dataset_root` at:

```text
tools2/data/fl_dataset_real
```

This is the final thesis dataset built from cleaned real RuralIoT measurements.
Before pushing a regenerated dataset, validate it locally:

```bash
cd tools2
python3 05_prepare_fl_dataset.py --force
python3 06_validate_fl_dataset.py
```

The old synthetic FL dataset is kept separately in `tools2/data/fl_dataset`.
