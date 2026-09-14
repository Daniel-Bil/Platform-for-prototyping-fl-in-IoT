"""Keras model construction compatible with the FL Builder/tools2 JSON graph format."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_model_config(path: str | Path) -> dict[str, Any]:
    """Load either supported model JSON schema and normalize it.

    Supported inputs:
      1. tools2/deployment wrapper: {"config": {...}, "architecture": {"nodes": [...], "edges": [...]}}
      2. FL Builder export:       {"name": "...", "nodes": [...], "edges": [...]}

    The rest of the deployment code always receives the wrapped form.
    """
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise ValueError("model JSON root must be an object")

    if isinstance(config.get("architecture"), dict):
        architecture = config["architecture"]
        if not isinstance(architecture.get("nodes"), list) or not isinstance(architecture.get("edges"), list):
            raise ValueError("'architecture' must contain 'nodes' and 'edges' arrays")
        return config

    # Native export produced by Daniel's FL Builder.
    if isinstance(config.get("nodes"), list) and isinstance(config.get("edges"), list):
        return {
            "name": config.get("name", Path(path).stem),
            "config": config.get("config", {}),
            "architecture": {
                "nodes": config["nodes"],
                "edges": config["edges"],
            },
        }

    raise ValueError(
        "model JSON must contain either an 'architecture' object or top-level 'nodes' and 'edges' arrays"
    )


def build_model_from_config(config: dict[str, Any], seq_len: int, num_features: int):
    # TensorFlow is imported lazily so networking/protocol tooling can run without it.
    from tensorflow.keras import layers, models

    architecture = config.get("architecture", {})
    raw_nodes = architecture.get("nodes", [])
    raw_edges = architecture.get("edges", [])
    if not raw_nodes:
        raise ValueError("model architecture contains no nodes")

    nodes = {node["id"]: node for node in raw_nodes}
    edges = [edge for edge in raw_edges if edge.get("isValid", True)]

    input_nodes = [node for node in nodes.values() if node.get("type") == "Input"]
    if len(input_nodes) != 1:
        raise ValueError(f"expected exactly one Input node, got {len(input_nodes)}")

    input_id = input_nodes[0]["id"]
    # Dataset shape is authoritative. The shape shown in the GUI JSON is metadata only.
    keras_input = layers.Input(shape=(seq_len, num_features), name=input_id)
    tensor_map = {input_id: keras_input}

    in_degree = {node_id: 0 for node_id in nodes}
    adjacency = {node_id: [] for node_id in nodes}
    incoming = {node_id: [] for node_id in nodes}

    for edge in edges:
        source, target = edge["from"], edge["to"]
        if source not in nodes or target not in nodes:
            raise ValueError(f"edge references unknown node: {source!r} -> {target!r}")
        adjacency[source].append(target)
        incoming[target].append(source)
        in_degree[target] += 1

    def recurrent_needs_sequences(node_id: str) -> bool:
        """Return True when an immediate downstream temporal layer needs a sequence.

        The common GUI pattern LSTM/GRU -> Dense should return a vector, not a full
        sequence. Stacked recurrent/Conv1D/pooling layers need sequence output.
        """
        temporal_consumers = {
            "LSTM", "GRU", "Conv1D", "Conv2D", "MaxPooling1D", "MaxPooling2D"
        }
        transparent = {"Dropout", "BatchNorm", "BatchNormalization", "ReLU", "LeakyReLU"}
        queue = list(adjacency[node_id])
        seen: set[str] = set()
        while queue:
            child = queue.pop(0)
            if child in seen:
                continue
            seen.add(child)
            child_type = nodes[child].get("type")
            if child_type in temporal_consumers:
                return True
            if child_type in transparent:
                queue.extend(adjacency[child])
        return False

    queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
    visited: list[str] = []

    while queue:
        current_id = queue.pop(0)
        visited.append(current_id)
        node = nodes[current_id]
        node_type = node.get("type")
        params = node.get("params", {})

        if current_id not in tensor_map:
            parent_tensors = [tensor_map[parent] for parent in incoming[current_id]]
            if not parent_tensors:
                raise ValueError(f"non-input node {current_id!r} has no input")
            x = parent_tensors[0] if len(parent_tensors) == 1 else parent_tensors

            if node_type in {"Conv1D", "Conv2D"}:
                x = layers.Conv1D(
                    filters=int(params.get("filters", 32)),
                    kernel_size=int(params.get("kernel_size", 3)),
                    activation=params.get("activation", "relu"),
                    padding=params.get("padding", "valid"),
                    name=current_id,
                )(x)
            elif node_type in {"MaxPooling1D", "MaxPooling2D"}:
                x = layers.MaxPooling1D(
                    pool_size=int(params.get("pool_size", 2)),
                    padding=params.get("padding", "same"),
                    name=current_id,
                )(x)
            elif node_type == "Flatten":
                x = layers.Flatten(name=current_id)(x)
            elif node_type == "Dense":
                if not adjacency[current_id]:
                    # Final classifier is fixed to the binary anomaly-detection task.
                    x = layers.Dense(1, activation="sigmoid", name=current_id)(x)
                else:
                    x = layers.Dense(
                        units=int(params.get("units", 32)),
                        activation=params.get("activation", "relu"),
                        name=current_id,
                    )(x)
            elif node_type == "Dropout":
                x = layers.Dropout(float(params.get("rate", 0.2)), name=current_id)(x)
            elif node_type in {"LSTM", "GRU"}:
                rnn_layer = layers.LSTM if node_type == "LSTM" else layers.GRU
                x = rnn_layer(
                    units=int(params.get("units", 32)),
                    return_sequences=recurrent_needs_sequences(current_id),
                    name=current_id,
                )(x)
            elif node_type == "Concatenate":
                if not isinstance(x, list) or len(x) < 2:
                    raise ValueError("Concatenate requires at least two incoming tensors")
                x = layers.Concatenate(name=current_id)(x)
            elif node_type in {"BatchNorm", "BatchNormalization"}:
                x = layers.BatchNormalization(name=current_id)(x)
            elif node_type == "ReLU":
                x = layers.ReLU(name=current_id)(x)
            elif node_type == "LeakyReLU":
                slope = float(params.get("negative_slope", params.get("alpha", 0.3)))
                # TensorFlow 2.14 uses `alpha`; newer Keras accepts negative_slope.
                x = layers.LeakyReLU(alpha=slope, name=current_id)(x)
            else:
                raise ValueError(f"unsupported layer type {node_type!r} at node {current_id!r}")

            tensor_map[current_id] = x

        for neighbor in adjacency[current_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(visited) != len(nodes):
        raise ValueError("model graph is cyclic or disconnected from the input")

    output_nodes = [node_id for node_id, children in adjacency.items() if not children]
    if len(output_nodes) != 1:
        raise ValueError(f"expected exactly one output node, got {len(output_nodes)}")

    model = models.Model(
        inputs=keras_input,
        outputs=tensor_map[output_nodes[0]],
        name="DeploymentFederatedModel",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
