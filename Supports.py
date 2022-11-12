import random
import numpy as np
import seaborn as sns
import scipy
from scipy.sparse import dia_matrix
import time
from datetime import datetime
import cv2
from seaborn import heatmap
import matplotlib.pyplot as plt
import os
import re
import imageio
from PIL import Image

X_list = []
iters = 0


def Matrix_real(N, Q, k):
    M = N + 1
    data1 = k * k * (1 + Q) - 4 * N * N
    data2 = np.ones(M).reshape(-1, )
    data2[0] = 0
    data2[1] = 2
    data2_plus = N * N * np.tile(data2, M)
    data2_minus = np.flipud(data2_plus)
    data3 = np.ones(M * M).reshape(-1, )
    data3[:M] = 0
    data3[M:2 * M] = 2
    data3_plus = N * N * data3
    data3_minus = np.flipud(data3_plus)
    data = (
        np.c_[
            data1,
            data2_minus,
            data2_plus,
            data3_minus,
            data3_plus]).transpose()
    offsets = np.array([0, -1, 1, -M, M])
    dia = dia_matrix((data, offsets), shape=(M * M, M * M))
    return dia


def Matrix_imag(N, k):
    M = N + 1
    matrix__ = np.ones((M, M))
    matrix__[0, 0] = matrix__[-1, 0] = matrix__[-1, -1] = matrix__[0, -1] = 2
    matrix__[1:-1, 1:-1] = 0
    data1 = 2 * k * matrix__.reshape(-1, ) * N
    data = (np.c_[data1]).transpose()
    offsets = np.array([0])
    dia = dia_matrix((data, offsets), shape=(M * M, M * M))
    return dia


#
#
# def Matrix_gen(N,Q,k):
#     M_r = Matrix_real(N,Q,k)
#     M_i = Matrix_imag(N,k)
#     M_i1 = -M_i
#     MM = np.array(np.bmat('M_r M_i1; M_i M_r'))
#     return MM


# def Matrix_Gen(N,Q,k):
#     tmp = Matrix_real(N,Q,k)+1j*Matrix_imag(N,k)
#     return tmp.tocoo()
#
def Matrix_Gen(N, Q, k):
    M = N + 1
    data1 = k * k * (1 + Q) - 4 * N * N
    data1 = np.tile(data1, 2)
    data2 = np.ones(M).reshape(-1, )
    data2[0] = 0
    data2[1] = 2
    data2 = np.tile(data2, 2)
    data2_plus = N * N * np.tile(data2, M)
    data2_minus = np.flipud(data2_plus)
    data3 = np.ones(M * M).reshape(-1, )
    data3[:M] = 0
    data3[M:2 * M] = 2
    data3_plus = N * N * data3
    data3_plus = np.tile(data3_plus, 2)
    data3_minus = np.flipud(data3_plus)
    matrix__ = np.ones((M, M))
    matrix__[0, 0] = matrix__[-1, 0] = matrix__[-1, -1] = matrix__[0, -1] = 2
    matrix__[1:-1, 1:-1] = 0
    data4 = -2 * k * matrix__.reshape(-1, ) * N
    data4_plus = np.tile(data4, 2)
    data4_minus = -data4_plus
    data = (
        np.c_[
            data1,
            data2_minus,
            data2_plus,
            data3_minus,
            data3_plus,
            data4_minus,
            data4_plus]).transpose()
    offsets = np.array([0, -1, 1, -M, M, -M * M, M * M])
    dia = dia_matrix((data, offsets), shape=(2 * M * M, 2 * M * M))
    return dia.tocoo()


def Error(a: np.ndarray, a_truth: np.ndarray) -> float:
    """
    Relative-Error
    """
    tmp = np.linalg.norm((a - a_truth), ord=2)
    return tmp / np.linalg.norm(a_truth, ord=2)


def Round(vector: np.ndarray, times: float) -> np.ndarray:
    SHAPE = vector.shape
    vector1 = vector.reshape(-1, )
    SHAPE1 = vector1.shape[0]
    radius = (vector.max()-vector.min())/2
    ERR = np.array([random.uniform(-radius * times, radius * times)
                    for i in range(SHAPE1)])
    return vector + ERR.reshape(SHAPE)


def Variation(N: int, q: np.ndarray) -> np.ndarray:
    """
    q.shape = ((N-1),(N-1))
    Returns.shape = ((N-1)**2,)
    """
    tmp = np.zeros((N + 1, N + 2))
    tmp[:, 0] = q[:, 0]
    tmp[:, -1] = -q[:, -1]
    tmp[:, 1:-1] = q[:, 1:] - q[:, :-1]
    return tmp


def TV(N: int, Q: np.ndarray) -> float:
    """
    Q.shape = ((N+1)**2,)
    """
    res = 0
    q = Q.reshape(N + 1, N + 1)
    res += abs(Variation(N, q)).sum()
    res += abs(Variation(N, q.transpose())).sum()
    return res


def TV_prime(N: int, Q: np.ndarray) -> np.ndarray:
    """
    Q.shape = ((N+1)**2,)
    """

    q = Q.reshape((N + 1, N + 1))
    res = np.zeros_like(q)

    def judge(q):
        return np.where(q < 0, -1, 1)
    res += np.c_[judge(q[:, :-1] - q[:, 1:]), judge(q[:, -1])]
    res += np.c_[judge(q[:, 0]), judge(q[:, 1:] - q[:, :-1])]
    q1 = q.T
    res += (np.c_[judge(q1[:, :-1] - q1[:, 1:]), judge(q1[:, -1])]).T
    res += (np.c_[judge(q1[:, 0]), judge(q1[:, 1:] - q1[:, :-1])]).T
    return res.reshape(-1,)


def callbackF(X):
    global X_list
    global iters
    X_list.append(X)
    iters += 1
    print('iter {} completed'.format(iters),
          '        %s' % str(datetime.now())[:-7])


def plot_heatmap(
        q_list,
        title,
        picdir,
        gifdir,
        subtitle_list,
        percent_list,
        pic_list):
    n = len(q_list)
    max_value = max([abs(qq).max() for qq in q_list])
    max_value = eval('%.2f' % max_value)
    img_list,img_list_tmp = [],[]
    for i in range(n):
        if i < n - 1:
            plt.figure(figsize=(4, 4))
            plt.title(str(subtitle_list[i]) + '    ' + percent_list[i])
            q = q_list[i]
            h = sns, heatmap(q, xticklabels=False, yticklabels=False,
                             cmap="gist_rainbow", vmin=-max_value, vmax=max_value, cbar=False)
            tmp_dir = picdir + title + '___' + str(i) + '.jpg'
            plt.savefig(tmp_dir)
            plt.close()
            img_list.append(cv2.imread(tmp_dir))
            img_list_tmp.append(cv2.imread(tmp_dir)[:, :, ::-1])
            os.remove(tmp_dir)
        else:
            plt.figure(figsize=(5, 4))
            plt.title(str(subtitle_list[i]) + '    ' + percent_list[i])
            q = q_list[i]
            h = sns, heatmap(q, xticklabels=False, yticklabels=False,
                             cmap="gist_rainbow", vmin=-max_value, vmax=max_value,cbar=True)
            tmp_dir = picdir + title + '___' + str(i) + '.jpg'
            plt.savefig(tmp_dir)
            plt.close()
            img_list.append(cv2.imread(tmp_dir))
            os.remove(tmp_dir)
            plt.figure(figsize=(4, 4))
            plt.title(str(subtitle_list[i]) + '    ' + percent_list[i])
            q = q_list[i]
            h = sns, heatmap(q, xticklabels=False, yticklabels=False,
                             cmap="gist_rainbow", vmin=-max_value, vmax=max_value, cbar=False)
            tmp_dir = picdir + title + '__' + str(i) + '.jpg'
            plt.savefig(tmp_dir)
            plt.close()
            img_list_tmp.append(cv2.imread(tmp_dir)[:, :, ::-1])
            os.remove(tmp_dir)
    imageio.mimsave(gifdir +title+'.gif', img_list_tmp, duration=0.5, loop=0)
    img_list1 = []
    for item in pic_list:
            img_list1.append(img_list[item])
    nn = len(img_list1)
    img1 = img_list1[0]
    for i in range(1, nn):
        img1 = np.hstack((img1, img_list1[i]))
    cv2.imwrite(picdir + title + '.jpg', img1)