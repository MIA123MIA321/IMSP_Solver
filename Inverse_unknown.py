from scipy.optimize import minimize
from Solver import *


def pdata_gen(N,Qt,k,f_list,matrix_A,solver = 'MUMPS'):
    tmp = []
    if solver == 'CG':
        for func in f_list:
            tmp.append(matrix_A @ mathscr_F0(N,Qt,k,Qt*func,solver))
    elif solver == 'MUMPS':
        Matrix_factorize(N, Qt, k)
        for func in f_list:
            tmp.append(matrix_A @ Matrix_solve(Qt*func))
    return tmp

def J(Q,N,partial_data,k,f_list,matrix_A,alpha = 0,solver = 'MUMPS'):
    tmp = []
    if solver == 'CG':
        for func in f_list:
            tmp.append(matrix_A @ mathscr_F0(N,Q,k,Q*func,solver))
    elif solver == 'MUMPS':
        Matrix_factorize(N, Q, k)
        for func in f_list:
            tmp.append(matrix_A @ Matrix_solve(Q*func))
    res = 0
    for i in range(len(f_list)):
        res += np.linalg.norm(tmp[i] - partial_data[i],ord = 2)**2
    return 0.5*k**(4-alpha)*res/len(f_list)


def J_prime(Q,N,partial_data,k,f_list,matrix_A,alpha = 0,solver = 'MUMPS'):
    res = np.zeros_like(Q)
    res_tmp = np.ones(((N+1),(N+1)))
    res_tmp[0,0] = res_tmp[-1,0] = res_tmp[0,-1] = res_tmp[-1,-1] = 0.5
    res_tmp[1:-1,1:-1] = 2
    res_tmp = res_tmp.reshape(-1,)
    if solver == 'CG':
        for j in range(len(f_list)):
            phii = mathscr_F0(N,Q,k,Q*f_list[j],solver)
            tmp = (f_list[j] - k * k * phii) * mathscr_F0(N,Q,k,matrix_A.T @ (matrix_A @ phii - partial_data[j]),solver)
            res += tmp[:(N+1)**2]+tmp[(N+1)**2:]
    elif solver == 'MUMPS':
        Matrix_factorize(N, Q, k)
        for j in range(len(f_list)):
            fr,fi = f_list[j].real,f_list[j].imag
            phi = Matrix_solve(Q*f_list[j])
            phi_r,phi_i = phi.real,phi.imag
            pd_r,pd_i = partial_data[j].real,partial_data[j].imag
            grad_r,grad_i = fr - k*k*phi_r,fi - k*k*phi_i
            tmpp = Matrix_solve(matrix_A.T @(matrix_A @ phi_r - pd_r))
            tmpr = grad_r*tmpp.real - grad_i*tmpp.imag
            tmpp = Matrix_solve(matrix_A.T @(matrix_A @ phi_i - pd_i))
            tmpi = grad_i * tmpp.real + grad_r * tmpp.imag
            res+= (tmpr+tmpi)*res_tmp
    return k**(4-alpha)*res/len(f_list)


def J_MULTI(Q,*argsargs):
    N, partial_data, k_list, f_list, matrix_A, alpha, solver, maxq = argsargs
    ans = 0
    for i in range(len(k_list)):
        ans += J(Q,N,partial_data[i],k_list[i],f_list[i],matrix_A,alpha,solver)
    return ans/maxq


def J_MULTIPRIME(Q,*argsargs):
    N, partial_data, k_list, f_list, matrix_A, alpha, solver, maxq = argsargs
    ans = np.zeros_like(Q)
    for i in range(len(k_list)):
        ans += J_prime(Q,N,partial_data[i],k_list[i],f_list[i],matrix_A,alpha,solver)
    return ans/maxq


def JTV_MULTI(Q,*args1):
    N, partial_data, k_list, f_list, matrix_A, alpha, solver, maxq, lamb = args1
    argsargs = N, partial_data, k_list, f_list, matrix_A, alpha, solver, maxq
    return lamb*0.5*TV(N,Q) + J_MULTI(Q,*argsargs)


def JTV_MULTIPRIME(Q,*args1):
    N, partial_data, k_list, f_list, matrix_A, alpha, solver, maxq, lamb = args1
    argsargs = N, partial_data, k_list, f_list, matrix_A, alpha, solver, maxq
    return lamb*0.5*TV(N,Q) + J_MULTIPRIME(Q,*argsargs)

def SOLVE(
        fun,
        Q0,
        args,
        jac,
        options={
            'disp': True,
            'gtol': 1e-5,
            'maxiter': 100},
        method='L-BFGS-B'):
    """
    args = N, q, k, arg_list, solver_for_this
    """
    if method == 'L-BFGS-B' or method == 'CG':
        res = minimize(
            fun,
            x0=Q0,
            args=args,
            method=method,
            jac=jac,
            options=options,callback=callbackF)
        return res.success, res.x


def RESULT(fun, Q0, Q_res, Q_truth, *arg_multi):
    return Error(Q0, Q_truth), Error(Q_res, Q_truth), fun(
        Q_truth, *arg_multi), fun(Q_res, *arg_multi)


if __name__ == '__main__':
    N = 5
    k_list = [1, 2, 3, 4]
    m = 5
    q = q_gen(N, 'Gauss')
    Q = q.reshape(-1, )
    matrix_A = gen_A(N)
    Matrix_analysis(N, Q, k_list[0])
    partial_data = []
    f_list = []
    for j in range(len(k_list)):
        tmpf = f_gen(N, k_list[j], m)
        f_list.append(tmpf)
        partial_data.append(pdata_gen(N, Q, k_list[j], tmpf, matrix_A, 'MUMPS'))
    k = k_list[2]
    f_ = f_list[2]
    partial_data = partial_data[2]
    print(J(Q, N, partial_data, k, f_, matrix_A, alpha=0, solver='MUMPS'))
    print(J_prime(Q*0, N, partial_data, k, f_, matrix_A, alpha=0, solver='MUMPS'))
    print('********************************')
    tmp = np.zeros((N+1)**2).reshape(-1,)
    tmpp = np.zeros((N+1)**2).reshape(-1,)
    print(Q.shape)
    print(tmp.shape)
    print('****************************************************************')
    for i in range((N+1)**2):
        tmp1 = np.zeros((N+1)**2).reshape(-1,)
        tmp1[i] = 0.00001
        tmp[i] = (J(tmpp+tmp1, N, partial_data, k, f_, matrix_A, alpha=0, solver='MUMPS') \
                 - J(tmpp, N, partial_data, k, f_, matrix_A, alpha=0, solver='MUMPS'))*10000
    print(tmp)

    ctx.destroy()