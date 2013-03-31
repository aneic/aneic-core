import pandas as pd
import numpy as np


def norm(data, axis=None): 
    if axis:
        return data / np.expand_dims(data.sum(axis), axis)  
    else: 
        return data / data.sum()

def nlog(x):
    '''Wraps log in nan_to_num'''
    return np.nan_to_num(np.log(x))

def convert_indicator(frame):
    '''Compute hierarchically indexed DataFrame with specified columns converted 
    to indicator variables (0 / 1). 
    '''
    cols = [get_indicator(frame[k]) for k in frame]
    return pd.concat(cols, axis=1, keys=frame.columns)


def get_indicator(series):
    """
    Convert categorical variable into dummy/indicator variables

    Parameters
    ----------
    series : Series

    Returns
    -------
    indicator : DataFrame
    """
    cat = pd.core.categorical.Categorical.from_array(np.asarray(series))
    dummy_mat = np.eye(len(cat.levels)).take(cat.labels, axis=0)
    dummy_mat[series.isnull()] = np.nan
    dummy_cols = cat.levels
    index = series.index
    return pd.DataFrame(dummy_mat, index=index, columns=dummy_cols)


def kl_gamma(g_a, g_b, eps=1e-2):
    '''
    Calculates difference between two sets of cluster assignments
    '''
    # ensure rows are aligned
    if g_a.shape[0] > g_b.shape[0]:
        g_a = g_a.reindex(g_b.index)
    if g_b.shape[0] > g_a.shape[0]:
        g_b = g_b.reindex(g_a.index)
    # cast as array
    g_a = np.asarray(g_a)    
    g_b = np.asarray(g_b)
    # pad columns if necessary
    if g_a.shape[1] > g_b.shape[1]:
        g_b = np.c_[g_b, np.zeros((g_b.shape[0], g_a.shape[1] - g_b.shape[1]))]
    if g_b.shape[1] > g_a.shape[1]:
        g_a = np.c_[g_a, np.zeros((g_a.shape[0], g_b.shape[1] - g_a.shape[1]))]
    # add small number to all entries to avoid sensitivity for zero values
    g_a = norm(g_a + eps, 1)
    g_b = norm(g_b + eps, 1)
    # element by element kl
    kl = lambda g_a, g_b: \
            g_a * nlog(g_a / g_b) + (1-g_a) * nlog((1-g_a) / (1-g_b))
    # full kl
    KL = lambda g_a, g_b: \
            (g_a * nlog(g_a / g_b)).sum(1)

    kl_ab = kl(g_a[:,:,None], g_b[:,None,:]).sum(0)
    kl_ba = kl(g_b[:,None,:], g_a[:,:,None]).sum(0)
    err = 0.5 * (kl_ab + kl_ba)

    # do greedy matching to minimize classification error
    perm = np.zeros((len(err),), 'i') - 1
    msk = (perm==-1)
    while msk.any():
        adxs = msk.nonzero()[0]
        bdxs = np.array(list(set(range(err.shape[1])) - set(perm)))
        k = err[adxs,:][:,bdxs].min(axis=1).argmin()
        l = err[adxs[k], bdxs].argmin()
        perm[adxs[k]] = bdxs[l]
        msk = (perm==-1)

    return 0.5 * (KL(g_a, g_b[:, perm]) + KL(g_b[:, perm], g_a)).mean()

def d_gamma(g_a, g_b, eps=0):
    '''
    Calculates rms distance between two cluster assignments
    '''
    # ensure rows are aligned
    if g_a.shape[0] > g_b.shape[0]:
        g_a = g_a.reindex(g_b.index)
    if g_b.shape[0] > g_a.shape[0]:
        g_b = g_b.reindex(g_a.index)
    # cast as array
    g_a = np.asarray(g_a)    
    g_b = np.asarray(g_b)
    # pad columns if necessary
    if g_a.shape[1] > g_b.shape[1]:
        g_b = np.c_[g_b, np.zeros((g_b.shape[0], g_a.shape[1] - g_b.shape[1]))]
    if g_b.shape[1] > g_a.shape[1]:
        g_a = np.c_[g_a, np.zeros((g_a.shape[0], g_b.shape[1] - g_a.shape[1]))]
    # add prior to all entries
    g_a = norm(g_a + eps, 1)
    g_b = norm(g_b + eps, 1)
    # element by element difference
    err = np.abs(g_a[:,:,None] - g_b[:,None,:]).sum(axis=0)

    # do greedy matching to minimize classification error
    perm = np.zeros((len(err),), 'i') - 1
    msk = (perm==-1)
    while msk.any():
        adxs = msk.nonzero()[0]
        bdxs = np.array(list(set(range(err.shape[1])) - set(perm)))
        k = err[adxs,:][:,bdxs].min(axis=1).argmin()
        l = err[adxs[k], bdxs].argmin()
        perm[adxs[k]] = bdxs[l]
        msk = (perm==-1)

    return 0.5 * np.abs(g_a - g_b[:, perm]).sum(axis=1).mean(axis=0)

def err_gamma(g_a, g_b, eps=0):
    '''
    Calculates rms distance between two cluster assignments
    '''
    # ensure rows are aligned
    if g_a.shape[0] > g_b.shape[0]:
        g_a = g_a.reindex(g_b.index)
    if g_b.shape[0] > g_a.shape[0]:
        g_b = g_b.reindex(g_a.index)
    # cast as array
    g_a = np.asarray(g_a)    
    g_b = np.asarray(g_b)
    # pad columns if necessary
    if g_a.shape[1] > g_b.shape[1]:
        g_b = np.c_[g_b, np.zeros((g_b.shape[0], g_a.shape[1] - g_b.shape[1]))]
    if g_b.shape[1] > g_a.shape[1]:
        g_a = np.c_[g_a, np.zeros((g_a.shape[0], g_b.shape[1] - g_a.shape[1]))]
    # add prior to all entries
    g_a = norm(g_a + eps, 1)
    g_b = norm(g_b + eps, 1)
    # calculate classification error
    # p(a,b) = a * (1-b) + (1-a) * b

    E = lambda a,b: a * (1-b) + (1-a) * b 
    err = E(g_a[:,:,None], g_b[:,None,:]).sum(axis=0)

    # do greedy matching to minimize classification error
    perm = np.zeros((len(err),), 'i') - 1
    msk = (perm==-1)
    while msk.any():
        adxs = msk.nonzero()[0]
        bdxs = np.array(list(set(range(err.shape[1])) - set(perm)))
        k = err[adxs,:][:,bdxs].min(axis=1).argmin()
        l = err[adxs[k], bdxs].argmin()
        perm[adxs[k]] = bdxs[l]
        msk = (perm==-1)

    return 0.5 * E(g_a, g_b[:, perm]).sum(axis=1).mean(axis=0)

def mi_gamma(g_a, g_b):
    '''
    Calculates mutual information and joint entropy of two
    sets of cluster assignements

        I(a,b) = E_p(a,b) [ log (p(a,b) / p(a)p(b)) ]
               = D_kl(p(a,b) || p(a)p(b))

        H(a,b) = - E_p(a,b) [ log p(a,b) ] 
    '''
    # align data if necessary
    if isinstance(g_a, pd.DataFrame):
        g_b = g_b.reindex(index=g_a.index)
    g_a = np.asarray(g_a)
    g_b = np.asarray(g_b)
    if (g_a.shape[1]==1) and (g_b.shape[1]==1):
        # size 1 edge case
        return 1., 1.
    else:
        # joint and marginals
        pab = (g_a[:, :, None] * g_b[:, None, :]).mean(0)
        pa = pab.sum(1)
        pb = pab.sum(0)
        Iab = (pab * nlog(pab / (pa[:,None] * pb[None,:]))).sum()
        Hab = - (pab * nlog(pab)).sum()
        return Iab, Hab


def vi_gamma(g_a, g_b):
    '''
    Calculates the variance of information. This is a mutual-
    information based distance metric for two sets of assignments

        d(g_a, g_b) = H(g_a, g_b) - I(g_a ; g_b)
    '''
    Iab, Hab = mi_gamma(g_a, g_b)
    return Hab - Iab


def vi0_gamma(g_a, g_b):
    '''
    Calculates the normalized variance of information. 

        d(g_a, g_b) = 1 - I(g_a ; g_b) / H(g_a, g_b)
    '''
    Iab, Hab = mi_gamma(g_a, g_b)
    return 1. - Iab / Hab


def _log_px_z(real, mu, l):
    '''calculate log probabilities for real-valued features'''
    log_norm = lambda x, mu, l: 0.5 * (np.log(0.5*l/np.pi) - l * (x - mu)**2)
    return pd.concat([log_norm(real, mu[k], l[k]).sum(1) 
                      for k in mu], axis=1)


def _log_py_z(cat, rho):
    '''calculate log probabilities for categorical features'''
    return pd.concat([((np.log(rho[k]) * cat).sum(1, level=0)).sum(1)
                      for k in rho], axis=1)


def _log_pxyz(real, cat, mu, l, rho, pi):
    '''calculate log joint'''
    # calculate mixture weights and log likelihood
    log_px_z = _log_px_z(real, mu, l)
    log_py_z = _log_py_z(cat, rho)
    return (log_px_z.fillna(0) + log_py_z.fillna(0)) + np.log(pi)


def _log_pq(mu, l, rho, pi, a=None, b=None, alpha=None, beta=None):
    import scipy.special as sp
    if (a is None) or (b is None):
        log_pl = pd.Series(np.zeros(mu.shape[1]), index=mu.columns)
    else:
        log_pl = (- sp.gammaln(a) 
                  + a * np.log(b) 
                  + (a - 1) * np.log(l) 
                  - b * l).sum(0)
    if alpha is None:
        log_pr = pd.Series(np.zeros(rho.shape[1]), index=rho.columns)
    else:
        log_pr = (sp.gammaln(alpha.sum(0, level=0)) 
                  + (np.log(rho) * (alpha - 1) 
                  - sp.gammaln(alpha)).sum(0, level=0)).sum(0)
    if beta is None:
        log_pz = 0.
    else:
        log_pz = (sp.gammaln(beta.sum()) 
                  + (np.log(pi) * (beta - 1) 
                  - sp.gammaln(beta)).sum()).sum()

    return log_pr + log_pl + log_pz

def e_step(real, cat, 
           pi, mu, l, rho, 
           a=None, b=None, alpha=None, beta=None):
    # caclulate responsibilities
    log_pxyz = _log_pxyz(real, cat, mu, l, rho, pi)
    pxyz = np.exp(log_pxyz)
    gamma = (pxyz.T / pxyz.sum(1)).T
    # calculate log likelihood
    g = gamma.as_matrix()
    L = np.nan_to_num(g * (log_pxyz.as_matrix() - np.log(g))).sum(1).sum(0)
    # calculate prior probabilities
    log_pq = _log_pq(mu, l, rho, pi, a, b, alpha, beta)
    return gamma, L + log_pq.sum(0)


def m_step(real, cat, 
           gamma, a=None, b=None, alpha=None, beta=None):
    # update for mu and l 
    X = pd.concat([(gamma[k].T * real.T).sum(1) for k in gamma], axis=1)
    X2 = pd.concat([(gamma[k].T * real.T**2).sum(1) for k in gamma], axis=1)
    mask = pd.notnull(real)
    G = pd.concat([(gamma[k].T * mask.T).sum(1) for k in gamma], axis=1)
    mu = X / G
    if (a is None) or (b is None):
        l = (X2 / G - mu**2)**(-1)
    else:
        # conjugate exponential form
        #
        # p(x | eta) = g(eta) exp(eta u(x))
        # p(eta | nu, chi) = f(n,chi) g(eta)^nu exp(eta chi)
        # 
        # eta = -l / 2
        # g(l) = (l / (2 pi))^(1/2) exp(-l m^2 / 2)
        # u(x) = x^2 - 2 mu x
        # nu = 2 (a - 1)
        # chi = 2 b - 2 (a-1) mu^2
        #
        # update
        #
        # dg(eta) / deta = -1/l + mu^2
        #                = -(chi + sum u(x)) / (N + nu)
        #
        # (1 / l[k]) = - [chi + sum_n g[n,k] u(x[n])] / (G[k] + nu) + mu^2
        nu = 2 * (a - 1)
        chi = 2 * b - nu * mu**2
        l = ((X2 - 2 * mu * X + chi) / (G + nu) + mu**2)**(-1)
        
        # conjugate exponential form
        
        # p(x | l) = g(l) exp(l u(x))
        # p(l | nu, chi) = f(n,chi) g(l)^nu exp(l chi)
        
        # g(l) = (l / (2 pi))^(1/2) exp(-l m^2 / 2)
        # u(x) = -x^2 / 2 + mu x
        # nu = 2 (a - 1)
        # chi = -b + (a-1) mu^2
        
        # update
        
        # dg(l) / dl = (1/l - mu^2) / 2
        #            = -(chi + sum u(x)) / (N + nu)
        
        # (1 / l[k]) = - 2 [chi + sum_n g[n,k] u(x[n])] / (G[k] + nu) + mu^2

        # nu = 2 * (a - 1)
        # chi = -b + (a-1) * mu**2
        # U = -0.5 * X2 + mu * X
        # l = (-2 * (chi + U) / (G + nu) + mu**2)**(-1)
    # update for rho 
    counts = pd.concat([(gamma[k].T * cat.T).sum(1) for k in gamma], axis=1)
    if not alpha is None:
        counts += alpha - 1
    rho = counts / counts.sum(level=0).reindex(counts.index, level=0)    
    # update for pi
    pi = gamma.sum(0)
    if not beta is None:
        pi += beta - 1
    pi /= pi.sum(1)
    return pi, mu, l, rho


def em(real, cat, K=None, u=None, eps=1e-5, max_iter=100, g0=None, theta0=None):
    if u is None:
        u = {'a': None, 'b': None, 'rho': None, 'z': None}
    if not theta0 is None:
        g0 = e_step(real, cat, 
                    theta0['pi'], theta0['mu'], theta0['l'], theta0['rho'], 
                    u['a'], u['b'], u['rho'], u['z'])[0]
    if g0 is None and (not K is None):
        # initialize mixture weights randomly
        g0 = pd.DataFrame(np.random.dirichlet(np.ones(K,), len(real)), index=real.index)
    else: 
        raise ValueError('One of K, g0, or theta0 must be specified.')
    # get first guess for params from random weights
    q_ = {};
    (q_['pi'], q_['mu'], q_['l'], q_['rho']) \
        = m_step(real, cat, g0, u['a'], u['b'], u['rho'], u['z'])
    # em-iteration loop
    it = 0
    L = []
    q = {}
    while True:
        # e-step: calculate posterior probability for states
        g, L_ = \
            e_step(real, cat, 
                   q_['pi'], q_['mu'], q_['l'], q_['rho'],
                   u['a'], u['b'], u['rho'], u['z'])
        # check for convergence
        dL = ((L_ - L[-1]) / np.abs(L[-1])) if it else np.nan
        if it > 1 and ((dL < eps) or it > max_iter):
            break
        # update L and q if not converged
        L.append(L_)
        q['pi'], q['mu'], q['l'], q['rho'] = \
            q_['pi'], q_['mu'], q_['l'], q_['rho']
        # m-step: update parameters based on mixture weights
        q_['pi'], q_['mu'], q_['l'], q_['rho'] = \
            m_step(real, cat, g, u['a'], u['b'], u['alpha'], u['beta'])
        it += 1
    return q, g, L, g0
