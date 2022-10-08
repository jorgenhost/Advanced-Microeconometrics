import numpy as np
from numpy import linalg as la
import pandas as pd
from tabulate import tabulate
import os
from typing import Optional, List, Tuple, Union, Iterable
from scipy import stats


def estimate( 
        y: np.ndarray, x: np.ndarray, transform='', N=None, T=None
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

    Returns:
        dict: A dictionary with the results from the ols-estimation.
    """
    
    b_hat = est_ols(y,x)
    residual = y-x@b_hat
    SSR = residual.T@residual # Fill in
    SST = (y - np.mean(y)).T@(y - np.mean(y))
    R2 = 1-SSR/SST # Fill in

    sigma, cov, se, deg_of_frees = variance(transform, SSR, x, N, T)
    t_values =  b_hat/se # Fill in
    
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
) -> pd.Series:
    
    '''
    Args:
        Results (dict): pass the results (dict) output from the estimate()-function
        var_labels (list): List of variable names used previously in our regression
        name (str): the name given to the pd.Series as output
    
    Returns:
        A pd.Series with variable names as index. NB! Add number of obs (N), time periods (T), regressors (K) and degrees of freedoms manually if appropriate.
        Ideally, pass one result at a time, and then merge with pandas later on.
    
    '''
    sig_levels = {0.05: '*', 0.01: '**', 0.001: '***'} #Set significance level for p-value
    
    deg_of_frees = results['deg_of_frees'] #Extract degrees of freedom from results dict

    beta = pd.Series(results['b_hat'].reshape(-1), index=var_labels).round(4) #Make series of our coeff
    se = pd.Series(results['se'].reshape(-1), index=var_labels).round(4) #Make series of standard errors
    t_stat = pd.Series(results['t_values'].reshape(-1), index=var_labels).round(4) #Make series of t-values
    p_val = pd.Series(
                stats.t.cdf(-np.abs(t_stat),df=deg_of_frees)*2, index=var_labels).round(4) #Make series of p-values, using the deg of freedoms
    
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
            temp_df.at[var_index]=str(temp_df.at[var_index])+sig_levels[0.001]
        elif temp_df.at[p_val_index]<0.01:
            temp_df.at[var_index]=str(temp_df.at[var_index])+sig_levels[0.01]
        elif temp_df.at[p_val_index]<=0.05:
            temp_df.at[var_index]=str(temp_df.at[var_index])+sig_levels[0.05]
        elif temp_df.at[p_val_index]>0.05:
            temp_df.at[var_index]=temp_df.at[var_index]
        temp_df.at[se_index]=f'({temp_df.at[se_index]})'
        
    temp_df = temp_df.drop('pt', level=1) #Remove our 'helper' p-values
    
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

def strict_exo_test(
    x: np.array,
    y: np.array,
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
        lead_var (np.array) : The variable to lead in FE regression
        year (np.array)     : Array of year variable
    
    Returns:
        Returns estimate of this regression
    
    '''

    # A lead matrix is basically an Identity matrix, 
    # but with the diagonal 'one to the right'

    lead = 1
    lead_mat = np.eye(T, k=lead)[:-1]
    
    #Lead of var, this means we lose the last year
    lead_out = perm(lead_mat, lead_var)
    last_year = np.unique(year).max()


    # Lead of emp var, this means we lose the last year
    #Let's trim our matrix x and vector y so we don't have the last year  
    x_exo = x[year!=last_year]
    y_exo = y[year!=last_year]

    # We then 'stack horizontally' our union_lead into our matrix
    # This adds a vector of observations - in this case our union_lead
    x_lead_FE = np.hstack((x_exo, lead_out))

    #Lets demean our y
    Q_T = demeaning_matrix(T-1) #Remember to subtract one time period
    y_lead_demean = perm(Q_T, y_exo)
    x_lead_demean = perm(Q_T, x_lead_FE)

    return estimate(y_lead_demean, x_lead_demean, transform='fe', T=T-1, N=N)

def demeaning_matrix(T):
    Q_T = np.eye(T)-np.tile(1/T,T)
    return Q_T