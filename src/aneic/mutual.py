# -*- coding: UTF-8 -*-
'''
Calculates mutual-information based quantities
'''
import numpy as np
import pandas as pd
from numpy import newaxis as na

# define normal distribution in terms of mean, mu and sigma
Norm = lambda x, mu, l: (l / (2*np.pi))**0.5 * np.exp(-0.5 * l * (x - mu)**2)

def Px_z(x, mu, l, bins=10):
    '''
    Calculate discretised probability distribution 
    for real-valued features
        
    px_z.ix[k,(ft,b)] = P(x[ft] = x[ft][b] | theta, z=k)
    
    Inputs
    ------  

    x : (N,R) pandas.DataFrame
        Real-valued features 
    mu : (K,R) pandas.DataFrame
        State means
    l : (K,R) pandas.DataFrame
        State precisions
    bins : int (default = 10)
        Number of bins to discretise real values into

    Outputs
    -------

    px_z : (K, R*B) pandas.DataFrame
        Discretized likelihood over B bins for each feature and 
        cluster component.
    '''
    # get dimensions
    K = len(mu)
    B = bins
    N, R = x.shape
    px_z = {}
    for ft in x:
        xh, xb = np.histogram(x[ft].dropna(), bins=B-1)
        mu_ = np.array(mu.ix[ft, :])
        l_ = np.array(l.ix[ft, :])
        px_z_ = Norm(xb[None, :], mu_[:, None], l_[:, None])
        px_z[ft] = pd.DataFrame(px_z_ / px_z_.sum(1)[:,None], 
                        index=mu.columns, columns=xb)
    return pd.concat(px_z.values(), keys=px_z, axis=1).T

def _MIdz(Pd_z, Pz):
    '''
    Helper function for calculating mutual information
    '''
    # calculate joint between data d and state z
    Pdz =(Pd_z * Pz)
    # calculate marginal of data d
    Pd = Pdz.sum(1)
    # calculate product of marginals
    PdPz = pd.DataFrame(np.array(Pd)[:,None] * np.array(Pz)[None,:], 
                index=Pd.index, columns=Pz.index)
    return (Pdz * np.log(Pdz / PdPz)).sum(1).sum(0,level=0)

def _Hdz(Pd_z, Pz):
    '''
    Helper function for calculating entropy
    '''
    # calculate joint between data d and state z
    Pdz =(Pd_z * Pz)
    return - (Pdz * np.log(Pdz)).sum(1).sum(0,level=0)

def MIdz(real, cat, mu, l, rho, pi, gamma, bins=10):
    '''
    Calculates Mutual-Information between feature values
    and cluster assignments.
    '''
    px_z = Px_z(real, mu, l, bins)
    py_z = rho
    pz = pi
    return _MIdz(px_z, pz), _MIdz(py_z, pz)


def MI0dz(real, cat, mu, l, rho, pi, gamma, bins=10):
    '''
    Calculates Mutual-Information between feature values
    and cluster assignments, normalized by joint entropy
    '''
    px_z = Px_z(real, mu, l, bins)
    py_z = rho
    pz = pi
    # calculate for real-valued features
    MIxz = _MIdz(px_z, pz)
    Hxz = _Hdz(px_z, pz)
    MI0xz = MIxz / Hxz
    # calculate for cat-valued features
    MIyz = _MIdz(py_z, pz)
    Hyz = _Hdz(py_z, pz)
    MI0yz = MIyz / Hyz
    return MI0xz, MI0yz


def Hdz(data, theta, gamma, bins=10):
    '''
    Calculates entropy of joint distribution over feature 
    values and cluster assignments.
    '''
    px_z, xb = Px_z(data.x, theta, bins)
    py_z = rho
    pz = pi
    return _Hdz(px_z, pz), _Hdz(py_z, pz)
