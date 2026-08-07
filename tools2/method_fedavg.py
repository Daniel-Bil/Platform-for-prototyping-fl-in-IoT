import numpy as np

def fedavg_aggregate(local_weights_list):
    """
    Standardowa agregacja FedAvg.
    Wyciąga średnią arytmetyczną z wag wszystkich lokalnych klientów.
    """
    return [np.mean(w_tuple, axis=0) for w_tuple in zip(*local_weights_list)]