import random

import tensorflow as tf

from logic2.weights_operations import list2np, dequantize_weights_int, simple_dequantize_floats


def simple_training(model, data, X_train, Y_train):
    print("simple_training")
    model.set_weights(list2np(data["weights"]))
    history = model.fit(X_train, Y_train, epochs=2)
    return history


def fedProx_training(model, data, dataset, mu=0.01):
    print("fedProx_training")
    # Get the initial global weights (from the server)
    global_weights = list2np(data["weights"])

    # Set the local model's weights to the global weights
    model.set_weights(global_weights)

    error = random.randint(4, len(dataset))
    results = {
        "loss": [],
        "accuracy": []
    }

    optimizer = model.optimizer

    # Perform training with the proximal term
    for idx, (batch_data, batch_labels) in enumerate(dataset):
        if idx == error - 1:
            break
        print(f"train batch nr {idx}")

        # Use GradientTape to manually calculate gradients
        with tf.GradientTape() as tape:
            # Forward pass: compute predictions and regular loss
            predictions = model(batch_data, training=True)
            loss = model.compiled_loss(batch_labels, predictions)  # Use model's compiled loss function (e.g., cross-entropy)

            # Compute the proximal term: penalty for weight deviation
            local_weights = model.trainable_variables
            prox_term = 0
            for local_w, global_w in zip(local_weights, global_weights):
                prox_term += tf.reduce_sum(tf.square(local_w - global_w))

            # Add the proximal term to the loss
            prox_penalty = (mu / 2) * prox_term
            total_loss = loss + prox_penalty

        # Compute gradients with respect to the total loss
        gradients = tape.gradient(total_loss, model.trainable_variables)

        # Apply gradients to update the model's weights
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        model.compiled_metrics.update_state(batch_labels, predictions)

        # Access the accuracy (or other metrics) results
        accuracy = model.compiled_metrics.metrics[0].result()  # Assuming

        # Save the results
        results["loss"].append(float(total_loss.numpy()))  # Convert the tensor to a numpy value
        results["accuracy"].append(float(accuracy.numpy()))

    return results, error


def fedPaq_int_training(model, data, X_train, Y_train):
    model.set_weights(dequantize_weights_int(data["weights"], data["params"]))
    history = model.fit(X_train, Y_train, epochs=2)
    return history


def fedPaq_float_training(model, data, X_train, Y_train):
    model.set_weights(simple_dequantize_floats(data["weights"]))
    history = model.fit(X_train, Y_train, epochs=2)
    return history