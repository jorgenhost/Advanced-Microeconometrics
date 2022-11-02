from scipy.stats import norm
import numpy as np 
from sklearn.linear_model import Lasso

def penalty_BCCH(X_tilde,y):
    
    (N,p) = X_tilde.shape
    c = 1.1
    alpha = 0.05

    yXscale = (np.max((X_tilde.T ** 2) @ ((y-np.mean(y)) ** 2) / N)) ** 0.5

    penalty_pilot =  ((2*c*norm.ppf(1-alpha/(2*p)))/np.sqrt(N)) * yXscale

    clf_pilot = Lasso(alpha=penalty_pilot/2)

    clf_pilot.fit(X_tilde,y)

    preds = clf_pilot.predict(X_tilde)

    eps =  y-preds
    epsXscale = (np.max((X_tilde.T ** 2) @ (eps ** 2) / N)) ** 0.5

    lambda_BCCH =  ((2*c*norm.ppf(1-alpha/(2*p)))/np.sqrt(N)) * epsXscale

    return lambda_BCCH

def penalty_BRT(X_tilde,y):
    sigma = np.std(y)
    (N,p) = X_tilde.shape
    c = 1.1
    alpha = 0.05

    penalty_BRT= ((2*c*sigma)/np.sqrt(N))*(norm.ppf(1-alpha/(2*p)))

    return penalty_BRT

## TO-DO ## Partialling out LASSO func

def part_out_LASSO(
    X_tilde: np.ndarray, 
    Z: np.ndarray, 
    d: np.ndarray, 
    y: np.ndarray, 
    penalty=''):

    """Takes np.arrays of controls Z, treatment d and outcome y (non-standardized). Remember to add whole standardized X_tilde array of both treatment and controls, a requirement for POL.
    

    Args:
        X_tilde (np.ndarray)    : The standardized  independent variable(s) of both treatment d and controls z_i. 
        Z (np.ndarray)          : Matrix of controls z_i (non-standardized).
        d (np.ndarray)          : Vector of treatment variable (non-standardized)
        y (np.ndarray)          : Vector of dependent/outcome variable (non-standardized)
        penalty (str)           : Specify penalty types

    Returns:
        POL-estimate, penalty_yz, penalty_dz, CI-POL, se_POL, N, p
        N=obs
        p=no. of regressors
    """

    if penalty == 'BRT':
        penalty_func = penalty_BRT
    elif penalty == 'BCCH':
        penalty_func = penalty_BCCH
    else:
        raise Exception('Invalid penalty type.')

    ### STEP 0 ###
    # Standardize treatment variable
    # d_tilde = standardize(d) (Only for post double LASSO?)

    # Standardize our candidate controls z_i
    Z_tilde = standardize(Z)

    ### STEP 1 ###
    # Calculate penalty, based on LASSOing outcome variable (y) on controls (z_i).
    penalty_yz = penalty_func(X_tilde=Z_tilde, y=y)
    clf_BRT_yz = Lasso(alpha=penalty_yz/2) #Divide by 2 as per Lasso()-function
    clf_BRT_yz.fit(Z_tilde, y)
    preds_yz = clf_BRT_yz.predict(Z_tilde)

    # Saving residuals
    res_yz = y-preds_yz

    ### STEP 2 ###
    # Calculate penalty, based on LASSOing non-standardized treatment (d) on controls (z_i)
    penalty_dz = penalty_func(X_tilde=Z_tilde, y=d)
    clf_BRT_dz = Lasso(alpha=penalty_dz/2)
    clf_BRT_dz.fit(Z_tilde, d)
    preds_dz = clf_BRT_dz.predict(Z_tilde)
    coefs_dz = clf_BRT_dz.coef_

    # Saving residuals
    res_dz = d-preds_dz

    ### STEP 3 ###
    # Calculating estimate of treatment effect
    numerator = np.sum(res_yz*res_dz)
    denominator = np.sum(res_dz**2)

    # Post Partialing Out LASSO estimate of treatment effect
    POL = (numerator/denominator).round(2)

    ### STEP 4 ###
    # Calculate variance
    # Lasso outcome variable y on X=((d,z_i))
    penalty_BRT_yx = penalty_func(X_tilde=X_tilde, y=y)
    clf_yx = Lasso(alpha=penalty_BRT_yx/2)
    clf_yx.fit(X_tilde, y)
    preds_yx = clf_yx.predict(X_tilde)

    coefs_POL_BRT = clf_yx.coef_

    #Save residuals
    res_yx = y-preds_yx

    #Use residuals to calculate variance
    (N,p)=X_tilde.shape
    numerator = np.sum(res_dz**2*res_yz**2)/N
    denominator = (np.sum(res_dz**2)/N)**2
    sigma2_POL = numerator/denominator

    # Use variance to calculate confidence interval
    q = norm.ppf(1-0.025)
    se_POL=np.sqrt(sigma2_POL/N).round(2)
    CI_POL = ((POL-q*se_POL).round(2), (POL+q*se_POL).round(2))

    return POL, penalty_yz.round(2), penalty_dz.round(2), CI_POL, se_POL, N, p

def standardize(X):
    X_tilde = (X-X.mean())/X.std()
    return X_tilde