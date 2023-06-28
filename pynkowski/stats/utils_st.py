'''This submodule contains utility functions that are useful for different statistics.'''
import numpy as np

def subsample_us(us, dus, iters=1_000):
    """Return the thresholds where MFs (except for v0) are computed before averaging within the bins 'dus'.

    Parameters
    ----------
    us : np.array
        The thresholds at which MFs have to be computed.

    dus : np.array
        The width of the bins associated to the thresholds 'us'.
        
    iters : int, optional
        the number of thresholds to consider within each bin.

    Returns
    -------
    us : np.array
        The sequence of thresholds where MFs are computed before averaging within each bin, with shape (us.shape, iters).
    
    """
    return np.vstack([np.linspace(u-du/2, u+du/2, iters) for u, du in zip(us, dus)])

def define_ubins(us, edges):
    """Return the bins for the computation of statistics. They are returned as (centre of bin, width of bin).

    Parameters
    ----------
    us : np.array
        The thresholds at which MFs have to be computed.

    edges : bool, optional
        If False (default), the given `us` is assumed to be an array of uniformly distributed thresholds, which are taken as the central values of the bins.
        If True, input `us` is assumed to be a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform distributions of thresholds. 
        In the latter case, the effective thresholds are the central value of the given bins.

    Returns
    -------
    us : np.array
        The central value of the bins.
        
    dus : np.array
        The width of the bins.
    
    """
    us = np.atleast_1d(us)
        
    if edges:
        dus = (us[1:]-us[:-1])
        us = (us[1:]+us[:-1])/2.
    else:
        if us.shape == (1,):
            dus = np.array([0.1])
        else:
            dus = (us[1]-us[0])*np.ones(us.shape[0])           
            if not (np.isclose(us[1:]-us[:-1], dus[0])).all():
                raise ValueError('The threshold distribution is not uniform. Please set `edges=True`.')
    return us, dus



__all__ = ["subsample_us", "define_ubins"]

__docformat__ = "numpy"

 