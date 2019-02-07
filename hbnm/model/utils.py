#! /usr/bin/python
import numpy as np
from scipy.special import erf
from scipy.stats import spearmanr

""" Auxiliary Functions """
def load_model_params():
    """ 
    Returns the model's synaptic parameters defined
    in synaptic.py as a python dictionary.
    """
    from params import synaptic
    params = clean_builtins(vars(synaptic))
    return params

def clean_builtins(my_dict):
    """
    Cleans the dictionaries
    """

    cleaned_dict = dict()
    for key, val in my_dict.items():
        if key[:2] != "__":
            cleaned_dict[key] = val
    return cleaned_dict

def prefix_keys(my_dict, prefix, sep="_"):
    """
    Adds prefix to each key 
    """
    prefixed_dict = dict()
    for key, val in my_dict.items():
        prefixed_dict[prefix + sep + key] = val
    return prefixed_dict

def cov_to_corr(cov, full_matrix=True):
    """ 
    Generate correlation matrix from covariance matrix.
        full_matrix: set True if cov is the full 2n x 2n covariance
        matrix and you wish to return the covariance matrix of
        the upper left quadrant (i.e. the E-E block)     
    """

    corr = None
    if full_matrix:
        nc = cov.shape[0] / 2
        cov_EE = cov[:nc, :nc]
    else:
        cov_EE = cov
    cov_ii = np.diag(cov_EE)
    norm2 = np.outer(cov_ii, cov_ii)
    try:
        corr = cov_EE / np.sqrt(norm2)
    except FloatingPointError:
        if np.min(norm2) < 0:
            msg = 'Trying to take sqrt of negative number! Check stability of system.'
            raise ValueError(msg)
    return corr

def fisher_z(r):
    """
    Perform Fisher z-tranform to the correlation matrix
    """
    return np.arctanh(r)

def subdiag(x):
    """
    Returns the upper diagonal of the matrix 
    """
    N = x.shape[0]
    return x[np.triu_indices(N, k=1)]


def linearize_map(x):
    """
    linearization function for T1w/T2w maps
    """
    return erf((x - np.mean(x)) / x.std() / np.sqrt(2))

def normalize_sc(x):
    """
    Normalizes the Structural Connectivity and removes the diagonal terms
    """
    for ii in range(x.shape[0]): x[ii, ii] = 0.0
    x /= x.max()
    return x

def vcorrcoef(X, y):
    """
    Calculates the Pearson correlation coefficient between the columns of the matrix X, 
    and the array y. Here, X is expected to be the upper diagonal terms of the empirical
    FC matrices of each subject, and y is expected to be the uppoer diaoonal terms of the
    model FC matrix.
    """
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    ym = np.mean(y)
    r_num = np.sum((X - Xm) * (y - ym), axis=1)
    r_den = np.sqrt(np.sum((X - Xm) ** 2, axis=1) * np.sum((y - ym) ** 2))
    r = r_num / r_den
    return r

def inter_indv_var(fc):
    N_regions = fc.shape[0]
    N_subjects = fc.shape[2]

    fc_flat = np.empty((N_regions, N_regions - 1, N_subjects))
    for s in xrange(N_subjects):
        for n in xrange(N_regions):
            index = np.concatenate((np.arange(n), np.arange(n + 1, N_regions)))
            fc_flat[n, :, s] = fc[n, index, s]
    fmv2 = np.array([subdiag(spearmanr(fc_flat[ii])[0]) for ii in xrange(N_regions)])
    return 1.0 - fmv2.mean(1)
