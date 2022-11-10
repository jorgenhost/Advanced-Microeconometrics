import numpy as np
from numpy import linalg as la
import LinearDynamic as lm

def sequential_instruments(x:np.ndarray, T:int):
    """Takes x, and creates the instrument matrix.

    Args:
        >> x (np.ndarray): The instrument vector that we will use to create a new
        instrument matrix that uses all possible instruments each period.
        >> T (int): Number of periods (in the original dataset, before removing
        observations to do first differences or lags). 

    Returns:
        np.ndarray: A (n*(T - 1), k*T*(T - 1)/2) matrix, that has for each individual
        have used all instruments available each time period.
    """

    n = int(x.shape[0]/(T - 1))
    k = x.shape[1]
    Z = np.zeros((n*(T - 1), int(k*T*(T - 1) / 2)))

    # Loop through all persons, and then loop through their time periods.
    # If first time period, use only that as an instrument.
    # Second time period, use the first and this time period as instrument, etc. 
    # Second last time period (T-1)

    # Loop over each individual, we take T-1 steps.
    for i in range(0, n*(T - 1), T - 1):
        # We make some temporary arrays for the current individual
        zi = np.zeros((int(k*T*(T - 1) / 2), T - 1))
        xi = x[i: i + T - 1]

        # j is a help variable on how many instruments we create each period.
        # The first period have 1 iv variable, the next have 2, etc.
        j = 0
        for t in range(1, T):
            zi[j: (j + t), t - 1] = xi[:t].reshape(-1, )
            j += t
        # It was easier to fill the instruments row wise, so we need to transpose
        # the individual matrix before we add it to the main matrix.
        Z[i: i + T - 1] = zi.T
    return Z
