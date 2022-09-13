 
import numpy as np

def get_μ(cls):
    """Compute the first derivative of the covariance function at the origin for a Gaussian field 
    defined on the sphere with angular power spectrum 'cls'.

    Parameters
    ----------
    cls : np.array
        The angular power spectrum of the Gaussian field.
    
    Returns
    -------
    μ : float
        The first derivative of the covariance function of a field at the origin.
    
    """
    cls = np.array(cls, dtype=float)
    ℓ = np.arange(cls.shape[0])
    cls /= np.sum(cls * (2.*ℓ+1.) / (4.*np.pi))
    μ = np.sum(cls * (2.*ℓ+1.) * ℓ*(ℓ+1.) / (8.*np.pi))
    return μ

def define_mu(cls,μ):
    """Return the first derivative of the covariance function at the origin for a field 
    computed accordingly to which input variable is given.

    Parameters
    ----------
    cls : np.array, optional
        The angular power spectrum of the field.
        Default : None

    μ : scalar, optional
        The first derivative of the covariance function at the origin for the field. If None, μ=1 is assumed.
        Default : None

    Returns
    -------
    μ : scalar
        The first derivative of the covariance function of the field at the origin.

    Notes
    -----
    The derivatives are always computed full-sky regardless of input mask.
    
    """
    if (cls is not None) and (μ is not None):
        raise ValueError(r"Both cls and $\mu$ cannot be given")
    if (cls is not None) and (μ is None):
        return get_μ(cls)
    if (cls is None) and (μ is not None):
        return μ
    if (cls is None) and (μ is None):
        return 1.
    
def define_us_for_V(us,dus,iters=1000):
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
    us_ : np.array
        The sequence of thresholds where MFs are computed before averaging within each bin, with shape (us.shape, iters)
        
    iters : int, optional
        the number of thresholds considered within each bin.
    
    """
    return np.vstack([np.linspace(u-du/2, u+du/2, iters) for u, du in zip(us, dus)])




__all__ = ["get_μ", "define_mu", "define_us_for_V"]

__docformat__ = "numpy"

 
