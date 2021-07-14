import torch as tc
import numpy as np


def tc_matrixcorr(x, y):
    """
    return a correlation matrix between
    columns of x and columns of y.

    So, if X.size() == (1000,4) and Y.size() == (1000,5),
    then the result will be of size (4,5) with the
    (i,j) value equal to the pearsonr correlation coeff
    between column i in X and column j in Y
    """
    mean_x = tc.mean(x, 0)
    mean_y = tc.mean(y, 0)
    xm = x.sub(mean_x.expand_as(x))
    ym = y.sub(mean_y.expand_as(y))
    r_num = xm.t().mm(ym)
    r_den1 = tc.norm(xm,2,0)
    r_den2 = tc.norm(ym,2,0)
    r_den = r_den1.t().mm(r_den2)
    r_mat = r_num.div(r_den)
    return r_mat

def tc_flatten(x):
    return x.contiguous().view(-1)

def tc_iterproduct(*args):
    return tc.from_numpy(np.indices(args).reshape((len(args),-1)).T)

def tc_iterproduct_like(x):
    return tc_iterproduct(*x.size())
