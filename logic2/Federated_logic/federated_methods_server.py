import numpy as np
from scipy.optimize import linear_sum_assignment


def match_and_average_layer(client_weights, layer_idx):
    """
    Match and average neurons across clients for a specific layer.

    Parameters:
    - client_weights: List of weights from all clients.
    - layer_idx: Index of the current layer to process.

    Returns:
    - Averaged weights for the matched neurons for the layer.
    """
    # Get the weights of the current layer for all clients
    layer_weights = [weights[layer_idx] for weights in client_weights]

    # Check if the layer is 1D or 2D
    if len(layer_weights[0].shape) == 1:  # Handle 1D arrays (like biases)
        num_neurons = layer_weights[0].shape[0]

        # Initialize the cost matrix for matching neurons across clients
        cost_matrix = np.zeros((num_neurons, num_neurons))

        # Fill the cost matrix by calculating the distance between neurons across clients
        for i in range(num_neurons):
            for j in range(num_neurons):
                # Calculate distance between the first client's neuron and the others
                cost_matrix[i, j] = np.linalg.norm(layer_weights[0][i] - layer_weights[1][j])

    else:  # Handle 2D arrays (like weights in Dense layers)
        num_neurons = layer_weights[0].shape[-1]

        # Initialize the cost matrix for matching neurons across clients
        cost_matrix = np.zeros((num_neurons, num_neurons))

        # Fill the cost matrix by calculating the distance between neurons across clients
        for i in range(num_neurons):
            for j in range(num_neurons):
                # Calculate distance between the first client's neuron and the others
                cost_matrix[i, j] = np.linalg.norm(layer_weights[0][:, i] - layer_weights[1][:, j])

    # Use Hungarian algorithm (linear sum assignment) to match neurons/channels
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create an array to store the averaged neurons after matching
    averaged_layer_weights = np.zeros_like(layer_weights[0])

    # Perform the matching and averaging across all clients
    if len(layer_weights[0].shape) == 1:  # For 1D layers
        for i, j in zip(row_ind, col_ind):
            averaged_layer_weights[i] = np.mean([client_weights[c][layer_idx][j] for c in range(len(client_weights))],
                                                axis=0)
    else:  # For 2D layers
        for i, j in zip(row_ind, col_ind):
            averaged_layer_weights[:, i] = np.mean(
                [client_weights[c][layer_idx][:, j] for c in range(len(client_weights))], axis=0)

    return averaged_layer_weights

def aggregate_weights_fedma(client_weights):
    """
    Aggregate model weights from different clients using neuron matching and averaging.

    Parameters:
    - client_weights: List of weights from all clients [weights1, weights2, ..., N].

    Returns:
    - Global weights after aggregation.
    """
    # Number of layers in the model (assuming all clients have the same architecture)
    num_layers = len(client_weights[0])

    # Initialize list to hold the global model's aggregated weights
    global_weights = []

    # Loop through each layer and match/average neurons across clients
    for layer_idx in range(num_layers):
        # Match and average the neurons for this layer
        averaged_layer = match_and_average_layer(client_weights, layer_idx)
        global_weights.append(averaged_layer)

    return global_weights