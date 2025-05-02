import numpy as np


def classify(a, x):
    """
    Classify a document given in the form of features a and parameter x

    Parameters:
        a: d-dimensional vector of features
        x: d-dimensional vector of parameters
    """
    return int(1 / (1 + np.exp(a.dot(x))) < 0.5)


def test_classify(A, b, x):
    """
    Returns proportion of correct classifications on the dataset (A, b) with parameter x

    Parameters:
        A: (n, d)-dimensional matrix of features
        b:
        x: d-dimensional vector of parameters
    """
    return sum(b[i, 0] == classify(a, x) for i, a in enumerate(A)) / A.shape[0]
