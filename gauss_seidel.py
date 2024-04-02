import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib import cm

def f(x, y):
    return math.sin(x*12) 

def split(A):
    N = A.shape[0]
    D = np.zeros([N, N])
    for i in range(N):
        D[i][i] = A[i][i]
    U = np.triu(A, k=1) 
    L = np.tril(A, k=-1)
    return L, D, U

def Gauss_Seidel(A, b, K):
    L, D, U = split(A)
    x = np.zeros(b.size)
    for i in range(K):
        x = -np.dot(np.dot(inv(L + D), U), x) + np.dot(inv(L + D), b)
    return x

def pic(L, N, U):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.linspace(0, L, N)
    Y = np.linspace(0, L, N)
    X, Y = np.meshgrid(X, Y)
    Z = np.reshape(U, (N, N))

    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.magma(norm(Z))

    surf = ax.plot_surface(X, Y, Z, facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    ax.zaxis.set_major_formatter('{x:.04f}')

    plt.show()

def Implicit_Establishing(A, b, tau):
    u = Gauss_Seidel(A, b, K)
    for i in range(b.shape[0]):
        b[i] = b[i] + u[i] / tau
    return u, b
    
L = 1.
h = 0.02
tau = 0.01
K = 10

N = int(L / h) + 1
NN = N*N

x = 0
y = 0

U = np.zeros(NN)
A = np.zeros([NN, NN])
b = np.zeros(NN)

for i in range(N):
    for j in range(N):
        n = i * N + j

        if (not i) or (not j) or (i == N - 1) or (j == N - 1):
            A[n][n] = 1
            b[n] = 0
        else:
            A[n][n] = -4 / h ** 2

            A[n][n + 1] = 1 / h ** 2
            A[n][n - 1] = 1 / h ** 2

            A[n][n + N] = 1 / h ** 2
            A[n][n - N] = 1 / h ** 2

            b[n] = -f(x, y)
        x += h
    x = 0
    y += h

for i in range(5):
    new_U, new_b = Implicit_Establishing(A, b, tau)
    b = new_b
    U = new_U

pic(L, N, U)
