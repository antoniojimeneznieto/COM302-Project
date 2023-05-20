import numpy as np


def channel(sent_signal):
    sent_signal = np.array(sent_signal)
    assert np.size(sent_signal) <= 200, "n must be <= 100"
    n = np.size(sent_signal) // 2
    x = sent_signal[0:2 * n]
    s = sum(x ** 2) / np.size(x)
    sigma = 1
    if s > 1:
        sigma = np.sqrt(s)
    Z = np.random.normal(0, sigma, size=(2 * n, 1))
    A = np.array([[11, 10], [10, 11]])
    B = np.kron(np.eye(n), A)
    Y = B.dot(x) + Z
    return Y
