"""
Utility functions for tc.Tensors
"""

import pickle
import random
import torch as tc


def tc_allclose(x, y):
    """
    Determine whether two torch tensors have same values
    Mimics np.allclose
    """
    return tc.sum(tc.abs(x-y)) < 1e-5


def tc_flatten(x):
    """Flatten tensor"""
    return x.contiguous().view(-1)

def tc_c_flatten(x):
    """
    Flatten tensor, leaving channel intact.
    Assumes CHW format.
    """
    return x.contiguous().view(x.size(0), -1)

def tc_bc_flatten(x):
    """
    Flatten tensor, leaving batch and channel dims intact.
    Assumes BCHW format
    """
    return x.contiguous().view(x.size(0), x.size(1), -1)


def tc_zeros_like(x):
    return x.new().resize_as_(x).zero_()

def tc_ones_like(x):
    return x.new().resize_as_(x).fill_(1)

def tc_constant_like(x, val):
    return x.new().resize_as_(x).fill_(val)

def tc_uniform(lower, upper):
    return random.uniform(lower, upper)


def tc_gather_nd(x, coords):
    x = x.contiguous()
    inds = coords.mv(tc.LongTensor(x.stride()))
    x_gather = tc.index_select(tc_flatten(x), 0, inds)
    return x_gather



def tc_pearsonr(x, y):
    """
    mimics scipy.stats.pearsonr
    """
    mean_x = tc.mean(x)
    mean_y = tc.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = tc.norm(xm, 2) * tc.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def tc_corrcoef(x):
    """
    mimics np.corrcoef
    """
    # calculate covariance matrix of rows
    mean_x = tc.mean(x, 1)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = tc.diag(c)
    stddev = tc.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    c = tc.clamp(c, -1.0, 1.0)

    return c



def tc_random_choice(a, n_samples=1, replace=True, p=None):
    """
    Parameters
    -----------
    a : 1-D array-like
        If a tc.Tensor, a random sample is generated from its elements.
        If an int, the random sample is generated as if a was tc.range(n)
    n_samples : int, optional
        Number of samples to draw. Default is None, in which case a
        single value is returned.
    replace : boolean, optional
        Whether the sample is with or without replacement
    p : 1-D array-like, optional
        The probabilities associated with each entry in a.
        If not given the sample assumes a uniform distribution over all
        entries in a.

    Returns
    --------
    samples : 1-D ndarray, shape (size,)
        The generated random samples
    """
    if isinstance(a, int):
        a = tc.arange(0, a)

    if p is None:
        if replace:
            idx = tc.floor(tc.rand(n_samples)*a.size(0)).long()
        else:
            idx = tc.randperm(len(a))[:n_samples]
    else:
        if abs(1.0-sum(p)) > 1e-3:
            raise ValueError('p must sum to 1.0')
        if not replace:
            raise ValueError('replace must equal true if probabilities given')
        idx_vec = tc.cat([tc.zeros(round(p[i]*1000))+i for i in range(len(p))])
        idx = (tc.floor(tc.rand(n_samples)*999)).long()
        idx = idx_vec[idx].long()
    selection = a[idx]
    if n_samples == 1:
        selection = selection[0]
    return selection


def save_transform(file, transform):
    """
    Save a transform object
    """
    with open(file, 'wb') as output_file:
        pickler = pickle.Pickler(output_file, -1)
        pickler.dump(transform)


def load_transform(file):
    """
    Load a transform object
    """
    with open(file, 'rb') as input_file:
        transform = pickle.load(input_file)
    return transform
    


    
