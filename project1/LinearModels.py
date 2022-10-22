import numpy as np
from numpy import linalg as la
import pandas as pd
from tabulate import tabulate
import os
from typing import Optional, List, Tuple, Union, Iterable
from scipy.stats import chi2, t


def estimate( 
        y: np.ndarray, 
        x: np.ndarray, 
        transform='', 
        N=None, 
        T=None,
        robust_se=False
    ) -> dict:
    """Takes some np.arrays and estimates regular OLS, FE or FD.
    

    Args:
        y (np.ndarray): The dependent variable, needs to have the shape (n*t, 1)
        x (np.ndarray): The independent variable(s). If only one independent 
        variable, then it needs to have the shape (n*t, 1).
        transform (str, optional): Specify if estimating fe or fd, in order 
        to get correct variance estimation. Defaults to ''.
        N (int, optional): Number of observations. If panel, then the 
        number of individuals. Defaults to None.
        T (int, optional): If panel, then the number of periods an 
        individual is observerd. Defaults to None.
        robust_se (bool): Use robust std errors. Default is False (use normal standard errors)

    Returns:
        dict: A dictionary with the results from the ols-estimation.
    """
    
    b_hat = est_ols(y,x)
    residual = y-x@b_hat
    SSR = residual.T@residual # Fill in
    SST = (y - np.mean(y)).T@(y - np.mean(y))
    R2 = 1-SSR/SST # Fill in

    sigma, cov, se, deg_of_frees = variance(transform, SSR, x, N, T)
    if robust_se:
        cov, se = robust(x, residual, T)
    
    t_values = b_hat/se

    names = ['b_hat', 'se', 'sigma', 't_values', 'R2', 'cov', 'N', 'T', 'deg_of_frees']
    results = [b_hat, se, sigma, t_values, R2, cov, N, T, deg_of_frees]
    return dict(zip(names, results))

    
def est_ols( y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Estimates OLS using input arguments.

    Args:
        y (np.ndarray): Check estimate()
        x (np.ndarray): Check estimate()

    Returns:
        np.array: Estimated beta hats.
    """
    return la.inv(x.T@x)@(x.T@y)

def variance(
        transform: str, 
        SSR: float, 
        x: np.ndarray, 
        N: int,
        T: int
    ) -> tuple :
    """Use SSR and x array to calculate different variation of the variance.

    Args:
        transform (str): Specifiec if the data is transformed in any way.
        SSR (float): SSR
        x (np.ndarray): Array of independent variables.
        N (int, optional): Number of observations. If panel, then the 
        number of individuals. Defaults to None.
        T (int, optional): If panel, then the number of periods an 
        individual is observerd. Defaults to None.

    Raises:
        Exception: [description]

    Returns:
        tuple: [description]
    """

    K=x.shape[1]

    if transform in ('', 're', 'fd'):
        sigma = SSR/(N*T-K) # Fill in
        deg_of_frees = N*T-K
    elif transform.lower() == 'fe':
        sigma = SSR/(N*(T-1)-K) # Fill in
        deg_of_frees = N*(T-1)-K
    elif transform.lower() in ('be'): 
        sigma = SSR/(N-K) # Fill in
        deg_of_frees = N-K
    else:
        raise Exception('Invalid transform provided.')
    
    cov =  sigma*la.inv(x.T@x) # Fill in
    se =  np.sqrt(cov.diagonal()).reshape(-1, 1) # Fill in
    return sigma, cov, se, deg_of_frees


def print_table(
        labels: tuple,
        results: dict,
        headers=["", "Beta", "Se", "t-values"],
        title="Results",
        tablefmt=None,
        **kwargs
    ) -> None:
    label_y, label_x = labels
    # Create table for data on coefficients
    table = []
    for i, name in enumerate(label_x):
        row = [
            name, 
            results.get('b_hat')[i], 
            results.get('se')[i], 
            results.get('t_values')[i]
        ]
        table.append(row)
    
    # Print table
    print(title)
    print(f"Dependent variable: {label_y}\n")
    print(tabulate(table, headers, tablefmt, **kwargs))
    
    # Print data for model specification
    print(f"R\u00b2 = {results.get('R2').item():.3f}")
    print(f"\u03C3\u00b2 = {results.get('sigma').item():.3f}")
    print(f"DF={results.get('deg_of_frees')}")
    print(f"N={results.get('N')}")
    print(f"T={results.get('T')}")


def perm( Q_T: np.ndarray, A: np.ndarray, t=0) -> np.ndarray:
    """Takes a transformation matrix and performs the transformation on 
    the given vector or matrix.

    Args:
        Q_T (np.ndarray): The transformation matrix. Needs to have the same
        dimensions as number of years a person is in the sample.
        
        A (np.ndarray): The vector or matrix that is to be transformed. Has
        to be a 2d array.

    Returns:
        np.array: Returns the transformed vector or matrix.
    """
    # We can infer t from the shape of the transformation matrix.
    if t==0:
        t = Q_T.shape[1]

    # Initialize the numpy array
    Z = np.array([[]])
    Z = Z.reshape(0, A.shape[1])

    # Loop over the individuals, and permutate their values.
    for i in range(int(A.shape[0]/t)):
        Z = np.vstack((Z, Q_T@A[i*t: (i + 1)*t]))
    return Z

def check_rank(x: np.ndarray) -> str:
    """Takes a np.ndarray (matrix) and returns the rank.

    Args:
        x (np.ndarray): The matrix in question.

    Returns:
        string with result of rank condition.
    """
    rank = np.linalg.matrix_rank(x)
    if rank < x.shape[1]:
        result = f'The matrix is NOT full rank with rank = {rank}. Eliminate linearly dependent columns.'
    elif rank==x.shape[1]: 
        result = f'The matrix is of full rank with rank = {rank}'
    return result

def outreg(
    results: dict,
    var_labels: list,
    name: str,
    robust_se = False,
) -> pd.Series:
    
    '''
    Args:
        Results (dict)      : pass the results (dict) output from the estimate()-function
        var_labels (list)   : List of variable names used previously in our regression
        name (str)          : the name given to the pd.Series as output
        robust_se (bool)    : If robust std. errors have been used, write in brackets []
    
    Returns:
        A pd.Series with variable names as index. NB! Add number of obs (N), time periods (T), regressors (K) and degrees of freedoms manually if appropriate.
        Ideally, pass one result at a time, and then merge with pandas later on.
        When merging, use 'outer' as method. In this case, it picks up all labels defined from different estimations
    
    '''
    sig_levels = {0.05: '*', 0.01: '**', 0.001: '***'} #Set significance level for p-value
    
    deg_of_frees = results['deg_of_frees'] #Extract degrees of freedom from results dict

    beta = pd.Series(results['b_hat'].reshape(-1), index=var_labels).round(2) #Make series of our coeff
    se = pd.Series(results['se'].reshape(-1), index=var_labels).round(3) #Make series of standard errors
    t_stat = pd.Series(results['t_values'].reshape(-1), index=var_labels).round(4) #Make series of t-values
    p_val = pd.Series(
                t.cdf(-np.abs(t_stat),df=deg_of_frees)*2, index=var_labels).round(4) #Make series of p-values, using the deg of freedoms
    
    temp_df = pd.concat((beta, se, p_val), axis=1) #concatenate above into dataframe, index is the varlabels
    temp_df.columns=['beta', 'se', 'pt'] #set column names to beta, se, pt (p-values)
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
        p_val_index=temp_df.index[k]
        if temp_df.at[p_val_index]<0.001:
            temp_df.at[var_index]=f'{temp_df.at[var_index]}'+sig_levels[0.001]
        elif temp_df.at[p_val_index]<0.01:
            temp_df.at[var_index]=f'{temp_df.at[var_index]}'+sig_levels[0.01]
        elif temp_df.at[p_val_index]<=0.05:
            temp_df.at[var_index]=f'{temp_df.at[var_index]}'+sig_levels[0.05]
        elif temp_df.at[p_val_index]>0.05:
            temp_df.at[var_index]=temp_df.at[var_index]
        
        #Write standard errors 
        if robust_se == True:
            temp_df.at[se_index]=f'[{temp_df.at[se_index]}]' #In brackes for robust std errors
        else:
            temp_df.at[se_index]=f'({temp_df.at[se_index]})' #In parentheses for 'normal' std errors
        
    
    # Remove our 'helper' p-values
    temp_df = temp_df.drop('pt', level=1)
    
    return temp_df

def serial_corr(
    y: np.array,
    x: np.array, 
    T: int,
    N: int,
    year: np.array):

    '''
    Args:
    NB! This is made for an FD-estimate
        y (np.array)    : Array of outcome variable
        x (np.array)    : Array of covariates
        T (int)         : Number of time periods
        N (int)         : Number of observations
        year (np.array) : Array of year variable
    
    Returns:
        Returns estimate of this regression
    
    '''

    b_hat = est_ols(y,x)
    e = y-x@b_hat
    fd_t=np.eye(T-1)
    fd_t=fd_t[:-1]
    e_l = perm(fd_t,e)
    reduced_year = year[year != np.unique(year).min()]      #Remove the first year
    e = e[reduced_year != np.unique(reduced_year).min()]    #Remove the second year
    # e_l are the lagged values of e.
    return estimate(e, e_l,N=N,T=T-2) #We lose two time periods

def strict_exo_FE(
    y: np.array,
    x: np.array,
    T: int,
    N: int,
    lead_var: np.array,
    year: np.array,
):

    '''
    Args:
    NB! Remove time-variant column(s) from covariates before proceeding
        y (np.array)        : Array of outcome variable
        x (np.array)        : Array of covariates (remove time-invariant covariates before proceeding)
        T (int)             : Number of time periods
        N (int)             : Number of observations
        lead_var (np.array) : The variable(s) to lead in FE regression (Wooldridge p.325)
        year (np.array)     : Array of year variable
    
    Returns:
        Returns estimate of this regression
    
    '''

    # Create lead matrix.
    # A lead matrix is basically an Identity matrix, 
    # but with the diagonal 'one to the right'
    lead = 1
    lead_mat = np.eye(T, k=lead)[:-1]

    # Create a list of arrays to loop through
    list_of_array = np.split(lead_var, lead_var.shape[1], axis=1)

    # Intiate empty list to lead each column (or variable)
    list_of_array_lead = []

    # Loop through the list of arrays and lead these variables.
    # Append to the empty list
    for i in list_of_array:
        list_of_array_lead.append(perm(lead_mat, i))
    
    # Horizontally stack the arrays into one multi-dimensional array
    lead_out = np.hstack(list_of_array_lead)

    # Get the last year, as we lose this period in the estimation
    last_year = np.unique(year).max()


    # Let's trim our matrix x and vector y so we don't have the last year  
    x_exo = x[year!=last_year]
    y_exo = y[year!=last_year]

    # Horizontally stack the arrays x_exo and the array with our leaded (?) variables
    x_lead_FE = np.hstack((x_exo, lead_out))

    # Lets demean our y
    Q_T = demeaning_matrix(T-1) #Remember to subtract one time period
    y_lead_demean = perm(Q_T, y_exo)
    x_lead_demean = perm(Q_T, x_lead_FE)

    return estimate(y_lead_demean, x_lead_demean, transform='fe', T=T-1, N=N)

def strict_exo_FD(
    y_diff: np.array,
    x_diff: np.array,
    T: int,
    N: int,
    w: np.array,
    year: np.array,
):

    '''
    Args:
        y_diff (np.array)   : Array of outcome variable (differenced)
        x_diff (np.array)   : Array of covariates (differenced)
        T (int)             : Number of time periods
        N (int)             : Number of observations
        w (np.array)        : The subset of x to include in the test (Wooldridge p.325)
        year (np.array)     : Array of year variable
    
    Returns:
        Returns estimate of this regression
    
    
    
    '''

    #Make list of array to loop through
    list_of_array = np.split(w, w.shape[1], axis=1)

    #Initiate empty list
    list_of_array_no_last_period = []

    #For each column (or variable) in the aforementioned list, append the column where the obs from the last year have been removed
    for i in list_of_array:
        list_of_array_no_last_period.append(i[year!=year.min()])
    
    #Horizontally stack these arrays into one multi-dimensional array
    #Using w as notation from Wooldridge
    w = np.hstack(list_of_array_no_last_period)

    #Stack these into one final multi-dimensional array for which we derive our results from
    x_diff = np.hstack((x_diff,w))

    return  estimate(y=y_diff, x=x_diff, transform='fd', T=T-1, N=N)

def robust( x: np.array, residual: np.array, T:int) -> tuple:
    '''Calculates the robust variance estimator 

    ARGS: 
        x           : Array of (N*T, K) regressors.
        residual    : (From results dict). 
        T           : number of time periods 
    Returns:
        cov, se     : Tuple of (1) asymptotic robust variance-covariance matrix and (2) heteroskedastic robust standard errors.
    '''
    # If only cross sectional, we can easily use the diagonal.
    if (not T) or (T==1):
        Ainv = la.inv(x.T@x)
        uhat2 = residual ** 2
        uhat2_x = uhat2 * x # elementwise multiplication: avoids forming the diagonal matrix (RAM intensive!)
        cov = Ainv @ (x.T@uhat2_x) @ Ainv
    
    # Else we loop over each individual.
    else:
        N = int(x.shape[0] / T)
        K = int(x.shape[1])
        emp = np.zeros((K, K)) # some empty array
        
        for i in range(N):
            idx_i = slice(i*T, (i+1)*T) # index values for individual i 
            omega = residual[idx_i]@residual[idx_i].T
            emp += x[idx_i].T@omega@x[idx_i]

        cov = la.inv(x.T@x)@emp@la.inv(x.T@x)
    
    se = np.sqrt(np.diag(cov)).reshape(-1, 1)
    return cov, se

def demeaning_matrix(T):
    Q_T = np.eye(T)-np.tile(1/T,T)
    return Q_T

def wald_test(
    R: np.array,
    beta_hat: np.array,
    r: np.array,
    Avar: np.array
):

    '''Calculates the robust Wald test statistics that converges in distribution to the Chi^2(Q) distribution with Q as degrees of freedom. NB! Careful of the correct array dimensions + defining R!
    
    Q = Rank(R)

    Args: 
        R               : QxK matrix of a linear hypothesis. 
        beta_hat        : Matrix containing the estimates of beta
        r               : The null hypothesis
        Avar            : Asymptotic variance of beta_hat's
    
    Returns:
        Returns p-value of the 
    '''

    chi_val = (R@beta_hat - r).T@la.inv(R@Avar@R.T)@(R@beta_hat-r)
    
    # Calculates our 'Q' that we use 
    Q = la.matrix_rank(R)

    # Calculate our p-value for our null to be true
    p_val = chi2.sf(chi_val.item(), Q)

    return p_val.round(2), round(chi_val.item(),2)