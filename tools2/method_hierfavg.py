import numpy as np

def hierfavg_edge_aggregate(local_weights_for_edge):
    """
    Agregacja na poziomie węzła brzegowego (Edge Server).
    Uśrednia modele od klientów przypisanych do tego konkretnego węzła.
    """
    return [np.mean(w_tuple, axis=0) for w_tuple in zip(*local_weights_for_edge)]

def hierfavg_cloud_aggregate(edge_weights_list):
    """
    Agregacja na poziomie chmury (Global Cloud).
    Uśrednia modele otrzymane od wszystkich węzłów brzegowych.
    """
    return [np.mean(w_tuple, axis=0) for w_tuple in zip(*edge_weights_list)]