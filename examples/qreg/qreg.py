import numpy as np 
import pandas as pd 

def q(theta, y, x, tau): 
    N,K = x.shape 
    assert theta.size == K

    # 0. unpacking 
    beta = theta

    # 1. the "check"-function
    check = lambda u, tau : (tau - (u<0)) * u 
    q = check(y - x@beta, tau)

    return q

def sim_data(theta, N, alpha): 
    '''sim_data: heteroscedasticity model specification
    Args
        theta: (K,) coefficients on x 
        N: sample size to be simulated 
        alpha: (K,) coefficients on x in heteroscedasticity-term 
    
    Returns (y,x)
    '''
    beta = theta 
    K = beta.size 
    assert K > 1, f'K>1 required'
    assert alpha.size == K, f'alpha must be same size as beta'

    # 1. regressors 
    oo = np.ones((N,1))
    xx = np.random.uniform(size=(N,K-1)) # uniform => ensures positivity
    x = np.hstack([oo, xx])

    # 2. error term 
    epsilon = np.random.normal(size=(N,))
    u = (x@alpha) * epsilon 

    # 3. outcome 
    y = x@beta + u

    return y,x

def sim_data_alt(theta, N, icdf): 
    '''sim_data_alt: functional specification
    Args
        icdf: inverse cdf function, e.g. lambda u : scipy.stats.chi2.ppf(u, df=1)
    
    Notes: 
        This implements that theta[0] (intercept) is constant over quantiles, 
        so that only the slope coefficient is varying. This implies that we 
        can compute the "theoretical" theta(tau) values as theta[1]*icdf(tau)
        (and the intercept just as theta[0].)
    '''
    K = theta.size
    assert callable(icdf)

    oo = np.ones((N,1))
    xx = np.random.uniform(size=(N,1)) + theta[0]
    x = np.hstack([oo, xx])

    tau = np.random.uniform(size=(N,1))
    beta2 = icdf(tau) * theta[1]
    beta1 = np.ones((N,1)) # first coefficient constant across quantiles 
    beta = np.hstack([beta1, beta2])

    y = np.sum(x * beta, axis=1)

    return y,x

def starting_values(y,x,tau:float) -> np.ndarray: 
    '''starting_values: 
    Args
        tau (float): quantile, in (0;1) 
    Returns 
        theta0 (np.ndarray): K-vector of parameters
    '''
    N,K = x.shape 
    assert y.ndim == 1, f'y must be 1-dimensional'
    assert np.isscalar(tau), f'tau must be scalar'
    assert (tau>0.) & (tau<1.), f'tau must be in (0;1)'

    theta0 = np.linalg.solve(x.T@x, x.T@y)

    return theta0
        
def discretize_into_percentile_bins(z, Np): 
    '''discretize_into_percentile_bins:
    Args
        z: N-array
        Np (int): no. bins 
    Returns
        bin_idx: N-array of integers taking values in range(Np)
            (each corresponding to a percentile bin)
    '''
    prcts = np.percentile(z, np.linspace(0,100,Np+1))
    prcts[-1] = np.inf # since we do xL <= x < xR, we need the last end point to be very high 
    bin_idx = np.digitize(z, prcts, right=False)
    cats = np.unique(bin_idx)
    if len(cats) < Np: 
        print(f'Warning: only {len(cats)} categories created for x: there must be at least one mass point. ')
    return bin_idx
    
import matplotlib.pyplot as plt 
def plot_data_and_predicted_against_x2(y,x,theta,k=1): 
    yhat = predict(theta, x)
    fig,ax = plt.subplots(); 
    ax.plot(x[:, k], y, '.', alpha=0.3, label='Data')
    ax.plot(x[:, k], yhat, '-', label='Predicted')
    return ax

def compute_percentiles_by_group(y, x, tau: float, Np: int): 
    assert x.ndim == 1
    assert y.ndim == 1 
    d = pd.DataFrame({'y': y, 'x': x})
    d['grp'] = discretize_into_percentile_bins(x, Np)
    f = lambda x : np.quantile(x, tau)
    g = d.groupby('grp').agg({'y': f, 'x': 'mean'})
    return g.y.values, g.x.values

def plot_cond_percentile(y, x, tau, Np, ax, **kwargs): 
    yy, xx = compute_percentiles_by_group(y, x[:, 1], tau, Np)
    ax.plot(xx, yy, 'o', label=f'tau={tau}', **kwargs)

def predict(theta, x): 
    beta = theta 
    return x@beta 



