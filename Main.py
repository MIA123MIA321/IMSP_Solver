from Inverse_unknown import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=50)
parser.add_argument('--k', type = str, default='2')
parser.add_argument('--m', type=int, default=3)
parser.add_argument('--maxq', type=float, default=0.1)
parser.add_argument('--q0_times', type=float, default=0.0)
parser.add_argument('--q_method', type=str, default='1')
parser.add_argument('--noise_level_model', type=float, default=0.0)
parser.add_argument('--noise_level_data', type=float, default=0.0)
parser.add_argument('--gtol', type=float, default=1e-10)
parser.add_argument('--maxiter', type=int, default=30)
parser.add_argument('--pic_list', type=int,nargs = '*', default=[0,1,2,4,8,-2,-1])
parser.add_argument('--title', type=str, default='tmp')
parser.add_argument('--picdir', type=str, default='pic/process_jpg/')
parser.add_argument('--gifdir', type=str, default='pic/process_gif/')
parser.add_argument('--output_filename', type=str, required=True)
parser.add_argument('--lamb', type=float, default=0.0)
args = parser.parse_args()


N, k, m, maxq, q0_times, q_method, noise_level_model, noise_level_data\
    = args.N, args.k, args.m, args.maxq, args.q0_times, args.q_method,\
    args.noise_level_model, args.noise_level_data
gtol, maxiter, pic_list = args.gtol, args.maxiter, args.pic_list
title, picdir, gifdir, output_filename = args.title, args.picdir, args.gifdir, args.output_filename
lamb = args.lamb

if isinstance(k,str):
    tmp_k = k.split(',')
    k = [float(eval(item)) for item in tmp_k]

q = q_gen(N, q_method)
q = q / q.max()
q = q * maxq
Q = q.reshape(-1, )
matrix_A = gen_A(N)
Matrix_analysis(N, Q, k[0])
partial_data = []
f_list = []
for j in range(len(k)):
    tmpf = f_gen(N, k[j], m)
    f_list.append(tmpf)
    tmp_data = pdata_gen(N, Q, k[j], tmpf, matrix_A, 'MUMPS')
    tmp_data1 = [Round(item, noise_level_data) for item in tmp_data]
    partial_data.append(tmp_data1)
Q0 = q0_times * Q
Q0 = Round(Q0, noise_level_model)
args1 = (N, partial_data, k, f_list, matrix_A, 0, 'MUMPS', maxq, lamb)


X_list.append(Q0)
t0 = time.time()
print('start time:%s' % str(datetime.now())[:-7])
RES2 = SOLVE(
    JTV_MULTI,
    Q0=Q0,
    args=args1,
    jac=JTV_MULTIPRIME,
    options={
        'disp': True,
        'gtol': gtol,
        'maxiter': maxiter},
    method='L-BFGS-B')


time_avg = (time.time() - t0) / len(X_list)
ll = len(X_list)
plot_list, label_list, Error_list = [], [], []
for j in range(ll):
    Error_list.append(Error(X_list[j], Q))
    plot_list.append(X_list[j].reshape((N + 1, N + 1)))
    label_list.append('Iter = ' + str(j))
plot_list.append(Q.reshape((N + 1, N + 1)))
label_list.append('Qt')


fp = open(output_filename, 'a+')
print('****************************************************************', file=fp)
print('****************************************************************', file=fp)
print('%s' % str(datetime.now())[:-7], file=fp)
print('N={},m={},k={}'.format(N, m, k), file=fp)
print('gtol={},maxiter={},lamb={}'.format(gtol, maxiter,lamb), file=fp)
print('q_method={},q0={}qt,maxq={}'.format(
    q_method, q0_times, maxq), file=fp)
print('noise_level_model={},noise_level_data={}'.format(
    noise_level_model, noise_level_data), file=fp)
print('total_iter={},t_avg={:.2f}'.format(len(X_list[1:]), time_avg), file=fp)
print('relative_model_error:', file=fp)
print(Error_list, file=fp)
percent_list = [str(round(Error_list[i]*100,2))+'%' for i in range(len(Error_list))]
percent_list[0] = ''
label_list[0] = 'Init'
percent_list.append('')
fp.close()
plot_heatmap(plot_list, title, picdir, gifdir, label_list,percent_list,pic_list)


ctx.destroy()
