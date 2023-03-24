

from scipy.special import comb, gamma

def __MF_prefactor(d,j):
    """Compute the prefactor in the definition of Minkowski Functionals. This factor multiplies the integral of the curvatures.

    Parameters
    ----------
    d : int
        The dimension of the ambient space (e.g., 2 for the sphere).
        
    j : int
        The index of the Minkowski functional (`j>=1`).
    
    Returns
    -------
    prefactor : float
        The prefactor in the definition of MFs.
    
    """
    return 1./ (comb(d,j) * j * np.pi**(j/2.) / gamma(j/2. +1.))
    

def V0(field, us, edges=False, verbose=True):
    """Compute the first Minkowski Functional, $V_0$, normalized by the volume of the space.

    Parameters
    ----------
    field : DataField or TheoryField
        Field on which to compute $V_0$, which can be a theoretical field or a data field.
        
    us : np.array
        The thresholds where $V_0$ is computed.
        
    edges : bool, optional
        If False (default), the given `us` is assumed to be an array of uniformly distributed thresholds, which are taken as the central values of the bins.
        If True, input `us` is assumed to be a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform distributions of thresholds. 
        In the latter case, the effective thresholds are the central value of the given bins.
        
    verbose : bool, optional
        If True (default), progress bars are shown for the computations on data.
    
    Returns
    -------
    V0 : np.array()
        The values of the first Minkowski Functional at the given thresholds.
    
    """
    us = np.atleast_1d(us)
    
    if isinstance(field, TheoryField):
        us = mf.theory.subsample_us(us, dus)
        try:
            return np.mean(field.V0(us), axis=1)
        except AttributeError:
            raise NotImplementedError(f"The theoretical expectation of V0 for {field.name} fields is not implemented. If you know an expression, please get in touch.")
        
    if isinstance(field, DataField):
        try:
            return field.V0(us, dus)
        except AttributeError:                
            stat = np.zeros_like(us)
            for ii in tqdm(np.arange(us.shape[0]), disable=not verbose):
                stat[ii] = np.mean((field.field > us[ii])[field.mask])
            return stat
        
    else:
        raise TypeError(f"The field must be either TheoryField or DataField (or a subclass).")
    

def V1(field, us, edges=False, verbose=True):
    """Compute the second Minkowski Functional, $V_1$, normalized by the volume of the space.

    Parameters
    ----------
    field : DataField or TheoryField
        Field on which to compute $V_1$, which can be a theoretical field or a data field.
        
    us : np.array
        The thresholds where $V_1$ is computed.
        
    edges : bool, optional
        If False (default), the given `us` is assumed to be an array of uniformly distributed thresholds, which are taken as the central values of the bins.
        If True, input `us` is assumed to be a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform distributions of thresholds. 
        In the latter case, the effective thresholds are the central value of the given bins.
        
    verbose : bool, optional
        If True (default), progress bars are shown for the computations on data.
    
    Returns
    -------
    V1 : np.array()
        The values of the second Minkowski Functional at the given thresholds.
    
    """
    us = np.atleast_1d(us)
    us, dus = define_ubins(us, edges)
    
    if isinstance(field, TheoryField):
        us = mf.theory.subsample_us(us, dus)
        try:
            return np.mean(field.V1(us), axis=1)
        except AttributeError:
            raise NotImplementedError(f"The theoretical expectation of V1 for {field.name} fields is not implemented. If you know an expression, please get in touch.") from None
        
    elif isinstance(field, DataField):
        try:
            return field.V1(us, dus)
        except AttributeError:
            if field.first_der is None:
                field.get_first_der()
                
            stat = np.zeros_like(us)
            grad_modulus = np.sqrt(np.sum((field.first_der)**2., axis=0))
            for ii in tqdm(np.arange(us.shape[0]), disable=not verbose):
                this_mask = (us[ii] + dus[ii]/2. > field.field) & (us[ii] - dus[ii]/2. <= field.field)
                stat[ii] = np.mean((grad_modulus*this_mask/dus[ii])[field.mask])
            return __MF_prefactor(field.dim, 1) * stat
        
    else:
        raise TypeError(f"The field must be either TheoryField or DataField (or a subclass).")
        
        
      


__all__ = ["V0", "V1", "V2", "general_curvature"]

__docformat__ = "numpy"

