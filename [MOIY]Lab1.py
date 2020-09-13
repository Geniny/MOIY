import numpy as np
def get_A(N):
    A = np.zeros((N, N), dtype=float)
    for i in range(0, N):
            A[i,:] = input().split(' ')

    return A

def get_B(N):
    B = np.zeros((N, N), dtype=float)
    for i in range(0, N):
        B[i, :] = input().split(' ')

    return B

def get_X(N):
    X = np.zeros(N, dtype=float)
    X[:] = input().split(' ')

    return X

def func(A, B, X, i):
    i = i - 1
    print(A)
    print(B)
    L = np.matmul(B, X)


    if L[i] == 0.:
        print("NO")
        exit(1)
    else:
        print("YES")
    L_ = np.copy(L)
    L_[i] = -1
    print('Вектор L`: ')
    print(L_)

    L__ = L_ * (-1/L[1])
    print('Вектор L^: ')
    print(L__)

    E = np.diag([1.]*A.shape[0])
    print('Матрица E: ')
    print(E)

    print('Матрица P: ')
    E[:,i] = L__[:,0]
    P = E
    print(P)

    print('Обратная матрица A`: ')
    A_r = np.matmul(P, B)
    print(A_r)

n, i = input().split(' ')
N = int(n)
I = int(i)
A = get_A(N)
B = get_B(N)
X = get_X(N)
func(A, B, X, I)