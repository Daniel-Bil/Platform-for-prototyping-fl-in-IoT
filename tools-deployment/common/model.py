"""Keras model construction compatible with tools2/fl_model*.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_model_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or "architecture" not in config:
        raise ValueError("model JSON must contain an 'architecture' object")
    return config


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
                    padding="same",
                    name=current_id,
                )(x)
            elif node_type in {"MaxPooling1D", "MaxPooling2D"}:
                x = layers.MaxPooling1D(
                    pool_size=int(params.get("pool_size", 2)),
                    padding="same",
                    name=current_id,
                )(x)
            elif node_type == "Flatten":
                x = layers.Flatten(name=current_id)(x)
            elif node_type == "Dense":
                if not adjacency[current_id]:
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
                    return_sequences=bool(adjacency[current_id]),
                    name=current_id,
                )(x)
            elif node_type == "Concatenate":
                if not isinstance(x, list) or len(x) < 2:
                    raise ValueError("Concatenate requires at least two incoming tensors")
                x = layers.Concatenate(name=current_id)(x)
            elif node_type == "ReLU":
                x = layers.ReLU(name=current_id)(x)
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
