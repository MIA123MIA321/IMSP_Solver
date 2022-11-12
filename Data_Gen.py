import random
import numpy as np
from typing import Tuple


def q_gen_T(
        N: int,
        q_value: float = 1,
        x1: float = 0.2,
        x2: float = 0.4,
        x3: float = 0.7,
        y1: float = 0.2,
        y2: float = 0.3,
        y3: float = 0.6,
        y4: float = 0.7) -> np.ndarray:
    """
    Returns.shape = (N+1,N+1)
    """
    q = np.zeros((N + 1, N + 1))
    q[int(x1 * N):int(x2 * N), int(y1 * N):int(y4 * N)] = q_value
    q[int(x2 * N):int(x3 * N), int(y2 * N):int(y3 * N)] = -q_value
    return q


def q_gen_Gauss(
        N: int,
        b1: float = 0.3,
        b2: float = 0.6,
        a1: float = 200,
        a2: float = 100,
        gamma: float = 1) -> np.ndarray:
    """
    q(x,y) = \\lambda \\exp (  -a1(x-b)^2   -a2(y-b2)^2    )
    Returns.shape = (N+1,N+1)
    """
    q = np.zeros((N + 1, N + 1))
    tmp = np.linspace(1, N - 1, N - 1) / N
    Y, X = np.meshgrid(tmp, tmp)
    q[1:-1, 1:-1] = gamma * \
        np.exp(-a1 * (X - b1) ** 2) * np.exp(-a2 * (Y - b2) ** 2)
    #     Y, X = np.meshgrid(-a1 * (tmp - b1) ** 2, -a2 * (tmp - b2) ** 2)
    #     q[1:-1, 1:-1] = gamma * np.exp(X) * np.exp(Y)
    qq = np.where(q > 1e-3, q, 0)
    qq = np.where(qq > 0.5, qq, -qq)
    return qq


def q_gen_1(N: int):
    q = np.zeros((N + 1, N + 1))
    tmp = np.linspace(1, N - 1, N - 1) / N
    Y, X = np.meshgrid(tmp, tmp)
    X, Y = 6 * X - 3, 6 * Y - 3
    q[1:-1, 1:-1] = 0.3 * (1 - X) ** 2 * np.exp(-X ** 2 - (Y + 1) ** 2) - \
                    (0.2 * X - X ** 3 - Y ** 5) * np.exp(-X ** 2 - Y ** 2) - 0.03 * np.exp(-(X + 1) ** 2 - Y ** 2)
    return q


def q_gen(N: int, method: str = 'T') -> np.ndarray:
    """
    Returns.shape = (N+1,N+1)
    """
    if method == 'T':
        return q_gen_T(N)
    elif method == 'Gauss':
        return q_gen_Gauss(N)
    elif method == '1':
        return q_gen_1(N)
    elif method == 'multi_gauss':
        tmpq = np.zeros((N + 1, N + 1))
        tmpq += q_gen_Gauss(N, 0.3, 0.6, 200, 300, 1)
        tmpq += q_gen_Gauss(N, 0.5, 0.3, 300, 400, 0.8)
        tmpq += q_gen_Gauss(N, 0.8, 0.5, 200, 500, 0.7)
        return tmpq
    print('q_method wrong')


def gen_A(N):
    tmp0 = np.zeros(((N-1)*4,(N+1)**2))
    tmp = np.zeros((N+1,N+1))
    tmp[0] = tmp[-1] = tmp[:,0] = tmp[:,-1] = 1
    tmp[0,0] = tmp[0,-1] = tmp[-1,0] = tmp[-1,-1] = 0
    b = np.where(tmp > 0)
    for i in range(4*(N-1)):
        tmp0[i,b[0][i]*(N+1)+b[1][i]] = 1
    return tmp0


def gen_A1(N):
    tmp = gen_A(N)
    tmp0 = np.zeros_like(tmp)
    return np.array(np.bmat('tmp tmp0;tmp0 tmp'))

def freal_gen(N,k,m):
    res = []
    for j in range(m):
        tmp = np.linspace(0,1,N+1)
        Y,X = np.meshgrid(tmp, tmp)
        tmpp = np.cos(k*(X*np.cos(2*np.pi*j/m)+Y*np.sin(2*np.pi*j/m))).reshape(-1,)
        res.append(tmpp)
    return res


def fimag_gen(N,k,m):
    res = []
    for j in range(m):
        tmp = np.linspace(0,1,N+1)
        Y,X = np.meshgrid(tmp, tmp)
        tmpp = np.sin(k*(X*np.cos(2*np.pi*j/m)+Y*np.sin(2*np.pi*j/m))).reshape(-1,)
        res.append(tmpp)
    return res


def f_gen(N,k,m):
    res = []
    for j in range(m):
        tmp = np.linspace(0,1,N+1)
        Y,X = np.meshgrid(tmp, tmp)
        tmpp1 = np.cos(k*(X*np.cos(2*np.pi*j/m)+Y*np.sin(2*np.pi*j/m))).reshape(-1,)
        tmpp2 = np.cos(k * (X * np.cos(2*np.pi * j / m) + Y * np.sin(2*np.pi * j / m))).reshape(-1, )
        res.append(tmpp1+1j*tmpp2)
    return res