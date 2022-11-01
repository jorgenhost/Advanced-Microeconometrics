from scipy.stats import norm
import numpy as np 
from sklearn.linear_model import Lasso

def penalty_BCCH(X_tilde,y):
    
    (N,p) = X_tilde.shape
    c = 1.1
    alpha = 0.05

    yXscale = (np.max((X_tilde.T ** 2) @ ((y-np.mean(y)) ** 2) / N)) ** 0.5

    penalty_pilot =  ((2*c*norm.ppf(1-alpha/(2*p)))/np.sqrt(N)) * yXscale

    clf_pilot = Lasso(alpha=penalty_pilot/2, fit_intercept=False)

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

def part_out_LASSO(X_tilde, Z_tilde, y, penalty=''):

    # USE BRT

    ### STEP 0 ###
    # Standardize initial GDP
    B_tilde_95_rule = standardize(B_las_95_rule)

    # Standardize our candidate controls z_i
    Z_tilde_95_rule = standardize(Z_las_95_rule)

    ### STEP 1 ###
    # Calculate penalty, based on LASSOing gdp_growth (y) on controls (z_i).
    # Lasso GDP-growth on controls z_i
    penalty_BRT_yz = penalty_BRT(X_tilde=Z_tilde_95_rule, y=y)
    clf_BRT_yz = Lasso(alpha=penalty_BRT_yz/2)
    clf_BRT_yz.fit(Z_tilde_95_rule, y)
    preds_yz = clf_BRT_yz.predict(Z_tilde_95_rule)

    # Saving residuals
    res_yz = y-preds_yz

    ### STEP 2 ###
    # Calculate penalty, based on LASSOing initial GDP on controls (z_I)
    # LASSO initial GDP on controls
    penalty_BRT_bz = penalty_BRT(X_tilde=Z_tilde_95_rule, y=B_las_95_rule)
    clf_BRT_bz = Lasso(alpha=penalty_BRT_bz/2)
    clf_BRT_bz.fit(Z_tilde_95_rule, B_las_95_rule)
    preds_bz = clf_BRT_bz.predict(Z_tilde_95_rule)
    coefs_bz = clf_BRT_bz.coef_

    # Saving residuals
    res_bz = B_las_95_rule-preds_bz

    ### STEP 3 ###
    # Calculating estimate
    numerator = np.sum(res_yz*res_bz)
    denominator = np.sum(res_bz**2)

    # Post Partialing Out LASSO estimate
    POL_BRT = (numerator/denominator).round(2)

    ### STEP 4 ###
    # Lasso GDP-growth on X=((beta,z_i))
    penalty_BRT_yx = penalty_BRT(X_tilde_poly_95, y=y)
    clf_yx = Lasso(alpha=penalty_BRT_yx/2)
    clf_yx.fit(X_tilde_poly_95, y)
    preds_yx = clf_yx.predict(X_tilde_poly_95)

    coefs_POL_BRT = clf_yx.coef_

    #Save residuals
    res_yx = y-preds_yx

    #Use residuals to calculate variance
    (N,p)=X_tilde_poly_95.shape
    numerator = np.sum(res_bz**2*res_yz**2)/N
    denominator = (np.sum(res_bz**2)/N)**2
    sigma2_POL_BRT = numerator/denominator

    # Use variance to calculate confidence interval
    q = norm.ppf(1-0.025)
    se_POL_BRT=np.sqrt(sigma2_POL_BRT/N).round(2)
    CI_POL_BRT = ((POL_BRT-q*se_POL_BRT).round(2), (POL_BRT+q*se_POL_BRT).round(2))
    return

def standardize(X):
    X_tilde = (X-X.mean())/X.std()
    return X_tilde