import numpy as np 
from scipy.stats import norm
from numpy import linalg as la

name = 'Tobit'

def q(theta, y, x): 
    return None # Fill in 

def loglikelihood(theta, y, x): 
    assert y.ndim == 1, f'y should be 1-dimensional'
    assert theta.ndim == 1, f'theta should be 1-dimensional'

    # unpack parameters 
    b = theta[:-1] # first K parameters are betas, the last is sigma 
    sig = np.abs(theta[-1]) # take abs() to ensure positivity (in case the optimizer decides to try it)
    
    phi= None # fill in

    Phi = None # fill in
    Phi = np.clip(Phi, 1e-8, 1.-1e-8)


    ll =  None # fill in, HINT: you can get indicator functions by using (y>0) and (y==0)

    return ll



def starting_values(y,x): 
    '''starting_values
    Returns
        theta: K+1 array, where theta[:K] are betas, and theta[-1] is sigma (not squared)
    '''
    N,K = None # fill in
    b_ols = None # fill in
    res = None # fill in
    sig2hat = None # fill in
    sighat = np.sqrt(sig2hat) # our convention is that we estimate sigma, not sigma squared
    theta0 = np.append(b_ols, sighat)
    return theta0 

def predict(theta, x): 
    '''predict(): the expected value of y given x 
    Returns E, E_pos
        E: E(y|x)
        E_pos: E(y|x, y>0) 
    '''
    sigma = np.abs(theta[-1]) # last element in x is sigma
    theta = theta[:-1] # First elements are coefficients, the last is sigma
    # Fill in 
    E = x@theta*norm.pdf((x@theta)/sigma)+sigma*norm.cdf((x@theta)/sigma)
    Epos = x@theta + sigma*mills_ratio((x@theta)/sigma)
    return E, Epos

def mills_ratio(z): 
    return norm.pdf(z) / norm.cdf(z)

def sim_data(theta, N:int): 
    b = theta[:-1]
    sig = theta[-1]
    K=b.size

    # FILL IN : x will need to contain 1s (a constant term) and randomly generated variables
    xx = np.random.normal(size=(N,K-1)) # 
    oo = np.ones((N,1)) # constant term 
    x  = np.hstack([oo,xx]) # full x matrix 

    mean = 0
    std_dev = sig
    eps = np.random.normal(loc=mean, scale=std_dev, size=(N,)) # fill in
    y_lat= x@b+eps # Latent index
    assert y_lat.ndim==1

    # Return y>0
    # That is: for each element in latent index, only return the value >0
    
    y = np.fmax(y_lat, 0) # fill in

    return y,x
