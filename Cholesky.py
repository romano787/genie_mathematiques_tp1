import numpy as np


def Cholesky(A):

    A = np.array(A, float)
    L = np.zeros_like(A)
    n, m = np.shape(A)

    for j in range(n):
        for i in range(j, n):

            if i == j:
                L[i, j] = np.sqrt(A[i, j] - np.sum(L[i, :j]**2))

            else:
                L[i, j] = (A[i, j] - np.sum(L[i, :j]*L[j, :j])) / L[j, j]

    return L


def SolveLU(L, U, b):

    L = np.array(L, float)
    U = np.array(U, float)
    b = np.array(b, float)

    n, m = np.shape(L)
    y = np.zeros(n)
    x = np.zeros(n)

    for i in range(n):
        y[i] = (b[i] - np.sum(np.dot(L[i, :i], y[:i]))) / L[i, i]

    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.sum(np.dot(U[i, i+1:n], x[i+1:n]))) / U[i, i]

    x = np.reshape(x, (n, 1))

    return x
