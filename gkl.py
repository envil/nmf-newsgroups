import numpy as np


def nmf_gkl(A, w, h):
    (N, M) = np.shape(A)
    (N, K) = np.shape(w)
    # WH = w @ h
    A_over_WH = A / (w @ h)
    # updating H
    for k in range(K):
        for j in range(M):
            h[k, j] *= sum(w[:, k] * A_over_WH[:, j]) / sum(w[:, k])
    for k in range(K):
        for i in range(N):
            w[i, k] *= sum(A_over_WH[i, :] * h[i, :]) / sum(h[k, :])
    w = w / np.sum(w, axis=0)
    return w, h


n = 4
m = 5
k = 2
A = np.ones((n, m))
W = np.random.rand(n, k)
H = np.random.rand(k, m)
nmf_gkl(A, W, H)
