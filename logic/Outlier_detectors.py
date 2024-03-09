from sklearn.ensemble import IsolationForest
import numpy as np


def outlier_detector(samples, algorithm = "IsolationForest"):
    if algorithm == "IsolationForest":
        pass
        handler = IsolationForest(n_estimators=10, warm_start=True)
        handler.fit(samples)
        predictions = handler.predict(samples)
    elif algorithm == "Local Outlier Factor":
        pass
    else:
        pass

    return predictions






if __name__ == "__main__":
    pass


