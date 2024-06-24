'''This submodule contains the functions to compute the density distribution (and total number) of local maxima and minima of a field.'''
import numpy as np
from .utils_st import subsample_us, define_ubins
from ..data.base_da import DataField
from ..theory.base_th import TheoryField

try:
    from tqdm.auto import tqdm
except:
    tqdm = lambda x, disable: x
    print('tqdm not loaded')

def maxima(field, us, edges=False, verbose=True):
    """Compute the density distribution of maxima and the total number of maxima of a field.

    Parameters
    ----------
    field : DataField or TheoryField
        Field on which to compute the maxima distribution, which can be a theoretical field or a data field.
        
    us : np.array
        The thresholds where the distribution is computed.
        
    edges : bool, optional
        If False (default), the given `us` is assumed to be an array of uniformly distributed thresholds, which are taken as the central values of the bins.
        If True, input `us` is assumed to be a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform distributions of thresholds. 
        In the latter case, the effective thresholds are the central value of the given bins.
        
    verbose : bool, optional
        If True (default), progress bars are shown for some computations on data.
    
    Returns
    -------
    maxima_dist : np.array()
        The density distribution of the maxima at the given thresholds.

    number_total : float
        The total number of maxima.
    
    """
    us = np.atleast_1d(us)
    
    if isinstance(field, TheoryField):
        us, dus = define_ubins(us, edges)
        us = subsample_us(us, dus)
        try:
            return np.nanmean(field.maxima_dist(us), axis=1), field.maxima_total()
        except AttributeError:
            raise NotImplementedError(f"The theoretical expectation of the maxima distribution for {field.name} fields is not implemented. If you know an expression, please get in touch.")
        
    if isinstance(field, DataField):
        try:
            max_list = field.maxima_list()
        except AttributeError:                
            raise NotImplementedError(f"The computation of the maxima distribution for {field.name} fields is not implemented. If you know a method, please get in touch.")

        if not edges:
            if us.shape == (1,):
                du = 0.1
            else:
                du = us[1] - us[0]
            us = np.hstack([us-du/2, us[-1]+du/2])

        return np.histogram(max_list, bins=us, density=True)[0], max_list.shape[0]
        
    else:
        raise TypeError(f"The field must be either TheoryField or DataField (or a subclass).")
    
def minima(field, us, edges=False, verbose=True):
    """Compute the density distribution of minima and the total number of minima of a field.

    Parameters
    ----------
    field : DataField or TheoryField
        Field on which to compute the minima distribution, which can be a theoretical field or a data field.
        
    us : np.array
        The thresholds where the distribution is computed.
        
    edges : bool, optional
        If False (default), the given `us` is assumed to be an array of uniformly distributed thresholds, which are taken as the central values of the bins.
        If True, input `us` is assumed to be a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform distributions of thresholds. 
        In the latter case, the effective thresholds are the central value of the given bins.
        
    verbose : bool, optional
        If True (default), progress bars are shown for the computations on data.
    
    Returns
    -------
    minima_dist : np.array()
        The density distribution of the minima at the given thresholds.

    number_total : float
        The total number of minima.
    
    """
    us = np.atleast_1d(us)
    
    if isinstance(field, TheoryField):
        us, dus = define_ubins(us, edges)
        us = subsample_us(us, dus)
        try:
            return np.nanmean(field.minima_dist(us), axis=1), field.minima_total()
        except AttributeError:
            raise NotImplementedError(f"The theoretical expectation of the minima distribution for {field.name} fields is not implemented. If you know an expression, please get in touch.")
        
    if isinstance(field, DataField):
        try:
            min_list = field.minima_list()
        except AttributeError:                
            raise NotImplementedError(f"The computation of the minima distribution for {field.name} fields is not implemented. If you know a method, please get in touch.")

        if not edges:
            if us.shape == (1,):
                du = 0.1
            else:
                du = us[1] - us[0]
            us = np.hstack([us-du/2, us[-1]+du/2])

        return np.histogram(min_list, bins=us, density=True)[0], min_list.shape[0]
        
    else:
        raise TypeError(f"The field must be either TheoryField or DataField (or a subclass).")
    


__all__ = ["maxima", "minima"]

__docformat__ = "numpy"

