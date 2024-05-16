from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

def outlier_detector(samples, algorithm = "IsolationForest"):
    if algorithm == "IsolationForest":
        pass
        handler = IsolationForest(n_estimators=10, warm_start=True)
        handler.fit(samples)
        predictions = handler.predict(samples)
    elif algorithm == "Local Outlier Factor":
        lof = LocalOutlierFactor(novelty=True)
        lof.fit(samples)
        predictions = lof.predict(samples)
    elif algorithm == "OCSVM":
        clf = OneClassSVM(gamma='auto').fit(samples)
        predictions = clf.predict(samples)
    elif algorithm == "eliptic":
        cov = EllipticEnvelope(random_state=0).fit(samples)
        predictions = cov.predict(samples)
    else:
        raise Exception("nuh hu")

    return predictions






if __name__ == "__main__":
    pass


