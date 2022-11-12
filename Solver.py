from Supports import *
from scipy.sparse.linalg import cg
from mumps import DMumpsContext
from Data_Gen import *

ctx = DMumpsContext()
ctx.set_silent()


def CG_method(N: int, Q: np.ndarray, k: float, F: np.ndarray) -> np.ndarray:
    """
    Q.shape = ((N-1)**2,1)
    F.shape = ((N-1)**2,1)
    Returns.shape = ((N-1)**2,)
    """
    res = cg(Matrix_Gen(N, Q, k), F)
    if res[1] == 0:
        return res[0]
    print('can not converge')


def Matrix_analysis(N: int, Q: np.ndarray, k: float) -> None:
    """
    Q.shape = ((N-1)**2,1)
    """
    global ctx
    _Matrix_ = Matrix_Gen(N, Q, k)
    ctx.set_shape(_Matrix_.shape[0])
    if ctx.myid == 0:
        ctx.set_centralized_assembled_rows_cols(
            _Matrix_.row + 1, _Matrix_.col + 1)
    ctx.run(job=1)


def Matrix_factorize(N: int, Q: np.ndarray, k: float) -> None:
    global ctx
    _Matrix_ = Matrix_Gen(N, Q, k)
    ctx.set_centralized_assembled_values(_Matrix_.data)
    ctx.run(job=2)
    return


def Matrix_solve(F: np.ndarray) -> np.ndarray:
    """
    Q.shape = ((N-1)**2,1)
    F.shape = ((N-1)**2,1)
    Returns.shape = ((N-1)**2,)
    """
    global ctx
    M = F.shape[0]
    F = np.append(F.real,F.imag)
    _Right_ = F
    x = _Right_.copy()
    ctx.set_rhs(x)
    ctx.run(job=3)
    tmp = x.reshape(-1, )
    return tmp[:M]+1j*tmp[M:]


def mathscr_F0(
        N: int,
        Q: np.ndarray,
        k: float,
        F: np.ndarray,
        solver: str = 'CG') -> np.ndarray:
    """
    Q.shape = ((N-1)**2,)
    F.shape = ((N-1)**2,)
    Returns.shape = ((N-1)**2,)
    """
    global ctx
    if solver == 'CG':
        return CG_method(N, Q, k, F)
    elif solver == 'MUMPS':
        Matrix_factorize(N, Q, k)
        return Matrix_solve(F)