import numpy as np
from scipy.stats import genextreme
import pandas as pd

def q(theta, y, x): 
    '''q: Criterion function, passed to estimation.estimate().
    Args. 
        theta: (K,) vector of parameters 
        x: (N,J,K) matrix of covariates 
        y: (N,) vector of outcomes (integers in 0, 1, ..., J)

    Returns
        (N,) vector. 
    '''
    return -loglikelihood(theta,y,x) # Fill in 

def q2(theta, y, x): 
    '''q: Criterion function, passed to estimation.estimate().
    Args. 
        theta: (K,) vector of parameters 
        y: NB!! OBSERVED market shares/conditional choice prob (N,J)

    Returns
        (N,) vector. 
    '''
    return -loglikelihood2(theta,y,x) # Fill in 

def starting_values(y, x): 
    '''starting_values(): returns a "reasonable" vector of parameters from which to start estimation
    Returns
        theta0: (K,) vector of starting values for estimation
    '''
    N,J,K=x.shape
    theta0 = np.zeros((K,)) # Fill in 
    return theta0

def util(theta, x, MAXRESCALE:bool=True): 
    '''util: compute the deterministic part of utility, v, and max-rescale it
    Args. 
        theta: (K,) vector of parameters 
        x: (N,J,K) matrix of covariates 
        MAXRESCALE (optional): bool, we max-rescale if True (the default)
    
    Returns
        v: (N,J) matrix of (deterministic) utility components
    '''
    assert theta.ndim == 1 
    N,J,K = x.shape 

    # deterministic utility 
    v = x@theta # Fill in 

    if MAXRESCALE: 
        # subtract the row-max from each observation
        # keepdims maintains the second dimension, (N,1), so broadcasting is successful
        v -=  v.max(axis=1, keepdims=True)
    
    return v 

def loglikelihood(theta, y, x): 
    '''loglikelihood()
    Args. 
        theta: (K,) vector of parameters 
        x: (N,J,K) matrix of covariates 
        y: (N,) vector of outcomes (integers in 0, 1, ..., J-1)
    
    Returns
        ll_i: (N,) vector of loglikelihood contributions
    '''
    assert theta.ndim == 1 
    N,J,K = x.shape 

    # deterministic utility 
    v = util(theta, x) # Fill in (use util function)

    # denominator 
    denom = np.exp(v).sum(axis=1) # Fill in
    assert denom.ndim == 1 # make sure denom is 1-dimensional so that we can subtract it later 

    # utility at chosen alternative 
    # Fill in evaluate v at cols indicated by y 

    v_i = v[range(N),y]

    ll_i = v_i-np.log(denom) # Fill in 

    assert ll_i.ndim == 1 # we should return an (N,) vector 

    return ll_i

def loglikelihood2(theta, y, x): 
    '''loglikelihood()
    Args. 
        theta: (K,) vector of parameters 
        x: (N,J,K) matrix of covariates 
        y: NB!! OBSERVED market shares/conditional choice prob (N,J)
    
    Returns
        ll_i: (N,) vector of loglikelihood contributions
    '''
    assert theta.ndim == 1 
    N,J,K = x.shape 

    ccp = choice_prob(theta, x)

    ll_i = np.sum(y*np.log(ccp),axis=1, keepdims=False) #Conducting element-wise multiplication and summing over columns (car alternatives)

    assert ll_i.ndim == 1 # we should return an (N,) vector 

    return ll_i


def choice_prob(theta, x):
    '''choice_prob(): Computes the (N,J) matrix of choice probabilities 
    Args. 
        theta: (K,) vector of parameters 
        x: (N,J,K) matrix of covariates 
    
    Returns
        ccp: (N,J) matrix of probabilities 
    '''
    assert theta.ndim == 1, f'theta should have ndim == 1, got {theta.ndim}'
    N, J, K = x.shape
    
    # deterministic utility 
    v = util(theta,x) # Fill in (using util())

    # denominator 
    denom = np.exp(v).sum(axis=1, keepdims=True) # axis=1, since I'm summing over columns, i.e. the "J"-cars
    assert denom.ndim == 2 # denom must be (N,1) so we can divide an (N,J) matrix with it without broadcasting errors

    # Conditional choice probabilites
    ccp = np.exp(v)/denom # np.exp(v) does max-rescaling - important!!
    
    return ccp


def sim_data(N: int, theta: np.ndarray, J: int) -> tuple:
    """Takes input values N and J to specify the shape of the output data. The
    K dimension is inferred from the length of theta. Creates a y column vector
    that are the choice that maximises utility, and a x matrix that are the 
    covariates, drawn from a random normal distribution.

    Args:
        N (int): Number of households.'
        J (int): Number of choices.
        theta (np.ndarray): The true value of the coefficients.

    Returns:
        tuple: y,x
    """
    K = theta.size
    
    # 1. draw explanatory variables 
    x = np.random.normal(size=(N,J,K)) # Fill in, use np.random.normal(size=())

    # 2. draw error term 
    uni = np.random.uniform(size=(N,J)) # Fill in random uniforms
    e =  genextreme.ppf(uni, c=0) # Fill in: use inverse extreme value CDF


    # 3. deterministic part of utility (N,J)
    v = x@theta # Fill in 

    # 4. full utility 
    u = v+e # Fill in 
    
    # 5. chosen alternative
    # Find which choice that maximises value: this is the discrete choice 
    y = np.argmax(u, axis=1) # Fill in, use np.argmax(axis=1)
    assert y.ndim == 1 # y must be 1D
    
    return y,x

def outreg(
    results: dict,
    var_labels: list,
    name: str,
) -> pd.Series:
    
    '''
    Args:
        Results (dict)      : pass the results (dict) output from the estimate()-function
        var_labels (list)   : List of variable names used previously in our regression
        name (str)          : the name given to the pd.Series as output
        se_type (str)       : If robust std. errors have been used, write in brackets []
    
    Returns:
        A pd.Series with variable names as index. NB! Add number of obs (N), time periods (T), regressors (K) and degrees of freedoms manually if appropriate.
        Ideally, pass one result at a time, and then merge with pandas later on.
        When merging, use 'outer' as method. In this case, it picks up all labels defined from different estimations
    
    '''
    sig_levels = {0.1: '*', 0.05: '**', 0.01: '***'} #Set significance level for p-value
    

    theta = pd.Series(results['theta_hat'].reshape(-1), index=var_labels).round(2) #Make series of our coeff
    se = pd.Series(results['se'].reshape(-1), index=var_labels).round(3) #Make series of standard errors
    t_stat = pd.Series(abs(results['t-stat'].reshape(-1)), index=var_labels).round(4) #Make series of t-values
    
    temp_df = pd.concat((theta, se, t_stat), axis=1) #concatenate above into dataframe, index is the varlabels
    temp_df.columns=['theta', 'se', 't_stat'] #set column names to beta, se, pt (p-values)
    temp_df=temp_df.stack() #Stack it so we make a multiindex
    temp_df.name=name #Have to name the series

    #Defining i, j, k
    # i: index position of our coeffs/var_labels
    # j: index position of our standard errors
    # k: index position of our p-values
    # -> loop through these in increments of 3 and add stars for significance levels

    for i,j,k in zip(range(0,len(temp_df-2),3), range(1, len(temp_df-1),3), range(2,len(temp_df),3)):

        var_index=temp_df.index[i]

        se_index=temp_df.index[j]

        t_stat_index=temp_df.index[k]

        if temp_df.at[t_stat_index]>2.576:
            temp_df.at[var_index]=f'{temp_df.at[var_index]}'+sig_levels[0.01]
        elif temp_df.at[t_stat_index]>=1.96:
            temp_df.at[var_index]=f'{temp_df.at[var_index]}'+sig_levels[0.05]
        elif temp_df.at[t_stat_index]>1.645:
            temp_df.at[var_index]=f'{temp_df.at[var_index]}'+sig_levels[0.1]
        else:
            temp_df.at[var_index]=f'{temp_df.at[var_index]}'

        temp_df.at[se_index]=f'({temp_df.at[se_index]})' #In parentheses for 'normal' std errors
        
    
    # Remove our 'helper' t-values
    temp_df = temp_df.drop('t_stat', level=1)
    
    return temp_df
