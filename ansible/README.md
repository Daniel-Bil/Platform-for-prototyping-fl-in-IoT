# Ansible deployment

The inventory hostnames match aliases from `~/.ssh/config`, so Ansible reuses
existing SSH users, keys, and `ProxyJump main` configuration.

GitHub cloning uses the SSH repository URL and forwarded local ssh-agent. No
private SSH key is copied to any remote machine.

## Connectivity

```bash
ansible all -i ansible/inventory.yml -m ping
```

## Provision main + first client

```bash
ansible-playbook \
  -i ansible/inventory.yml \
  ansible/playbooks/01-provision-repo.yml \
  --limit 'main:device-1'
```

The provisioning playbook:

1. installs Git/Python prerequisites,
2. clones the repository if absent,
3. fetches GitHub,
4. explicitly checks out `deployment`,
5. creates `.venv-deployment`,
6. installs `tools-deployment/requirements.txt`.

To provision every machine, omit `--limit`.

## Start 1 server + N clients

Specify how many dataset-backed clients should participate. Clients are selected
in inventory order, so `-e fl_client_count=3` starts `device-1` through `device-3`.

```bash
ansible-playbook \
  -i ansible/inventory.yml \
  ansible/playbooks/02-start-one-client.yml \
  -e fl_client_count=3
```

Examples:

```bash
# main + device-1
-e fl_client_count=1

# main + device-1..device-4
-e fl_client_count=4

# main + every currently dataset-backed client (device-1..device-7)
-e fl_client_count=7
```

Before starting the selected clients, the playbook stops any stale
`fl-thesis-client.service` on all devices. `device-8` is provisionable but is
automatically excluded until `fl_dataset_name` is assigned to it.


## Python compatibility

`tensorflow==2.14.1` does not provide a CPython 3.12 wheel. Provisioning therefore keeps Ansible on the host system Python but installs Python 3.10 for `.venv-deployment`. On Ubuntu the playbook enables `ppa:deadsnakes/ppa` to obtain Python 3.10. If an old `.venv-deployment` was created with Python 3.12, it is automatically removed and recreated with Python 3.10.
