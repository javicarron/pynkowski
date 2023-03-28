 
import numpy as np
from scipy.special import eval_hermitenorm, gamma, comb, factorial, gammainc
from scipy.stats import norm


def get_σ(cls):
    """Compute the variance of a field with an angular power spectrum 'cls'.

    Parameters
    ----------
    cls : np.array
        The angular power spectrum of the Gaussian field.
    
    Returns
    -------
    σ² : float
        The variance of the field.
    
    """
    cls = np.array(cls, dtype=float)
    ℓ = np.arange(cls.shape[0])
    return np.sum(cls * (2.*ℓ+1.) / (4.*np.pi))


def get_μ(cls):
    """Compute the first derivative of the covariance function at the origin for a Gaussian field 
    defined on the sphere with angular power spectrum 'cls', which are normalised to unit variance.

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

def flag(kj, k):
    """Compute flag coefficients.

    Parameters
    ----------
    ki : int
        The upper index of the flag coefficient.
        
    k : int
        The lower index of the flag coefficient.
    
    Returns
    -------
    flag : float
        The value of the float coefficient.
    
    """
    assert np.all(kj>=k), 'The first argument must be larger than the second.'
    return comb(kj, k) * omega(kj) / (omega(k) * omega(kj-k))
    
def omega(j):
    """Compute the volume of the $j$—dimensional ball, $\omega_j$.

    Parameters
    ----------
    j : int
        The dimension of the ball.
    
    Returns
    -------
    omega : float
        The volume of the $j$—dimensional ball.
    
    """
    return np.pi**(j/2.) / gamma(j/2. +1.)


def rho(k, us):
    """Compute the density functions $\rho_k(us)$ for Gaussian fields needed for the Gaussian Kinematic Formula.

    Parameters
    ----------
    k : int
        The order of the density function.
        
    us : np.array
        The thresholds where the density function is evaluated.
    
    Returns
    -------
    rho : np.array
        The density function evaluated at the thresholds.
    
    """
    if k==0:
        return 1. - norm.cdf(us)
    else:
        return 1. / (2. * np.pi)**(k/2.) * norm.pdf(us) * eval_hermitenorm(k-1, us)
    
def rho_Chi2(k, dof, us):
    """Compute the density functions $\rho_{k, \Chi^2}(us)$ for $\Chi^2_k$ fields (with $k$ degrees of freedom), needed for the Gaussian Kinematic Formula.

    Parameters
    ----------
    k : int
        The order of the density function.
        
    dof : int
        The degrees of freedom of the $\Chi^2$.
        
    us : np.array
        The thresholds where the density function is evaluated.
    
    Returns
    -------
    rho : np.array
        The density function evaluated at the thresholds.
    
    """
    if k==0:
        return 1.- (us >= 0.) * gammainc(dof/2., us/2.)
    else:
        factor = (us >= 0.) * us**((dof-k)/2.) * np.exp(-us/2.) / ((2.*np.pi)**(k/2.) * gamma(dof/2.) * 2.**((dof-2)/2.))
        summ = np.zeros_like(us)
        for l in np.arange(0, ((k-1)//2)+1):
            for m in np.arange(0,k-1-2*l+1):
                summ += (dof>= k - m - 2*l) * comb(dof-1, k-1-m-2*l) * (-1)**(k-1+m+l) * factorial(k-1) / (factorial(m) * factorial(l) * 2**l ) * us**(m+l)
        return factor*summ

def __prepare_lkc(dim, lkc_ambient):
    """Define the Lipschitz–Killing Curvatures of the ambient manifold as the default ones (unit volume and the rest are 0), or verify their consistency. 
    If no argument is given, it defaults to a default 2D space.

    Parameters
    ----------
    dim : int, optional
        The dimension of the ambient manifold.
        
    lkc_ambient : list, optional
        A list of the Lipschitz–Killing Curvatures of the ambient manifold. Its lenght must be `dim+1` if both arguments are given.
        
    Returns
    ----------
    dim : int 
        The dimension of the ambient manifold.
        
    lkc_ambient : np.array or None, optional
        An array of the Lipschitz–Killing Curvatures of the ambient manifold.
    """
    if lkc_ambient is None: 
        if dim is None:
            dim = 2
        lkc_ambient = np.zeros(dim+1)
        lkc_ambient[-1] = 1.
    else:
        if dim is None:
            dim = len(lkc_ambient) -1
        else:
            assert len(lkc_ambient) == dim +1, 'If both dim and lkc_ambient are given, len(lkc_ambient) == dim +1'
    return dim, lkc_ambient
    
def LKC(j, us, mu, dim=None, lkc_ambient=None):
    """Compute the expected value of the Lipschitz–Killing Curvatures (LKC) of the excursion set for Gaussian Isotropic fields.

    Parameters
    ----------
    j : int
        The index of the LKC.
        
    us : np.array
        The thresholds where the LKC is evaluated.

    mu : float
        The value of the derivative of the covariance function at the origin for the field.
        
    dim : int, optional
        The dimension of the ambient manifold.
        
    lkc_ambient : np.array, optional
        An array of the Lipschitz–Killing Curvatures of the ambient manifold. Its lenght must be `dim+1` if `dim` is also given.
        
    Returns
    ----------
    LKC : np.array
        The expected value of the Lipschitz–Killing Curvatures at the thresholds.
    """
    dim, lkc_ambient = __prepare_lkc(dim, lkc_ambient)
    result = np.zeros_like(us)
    for k in np.arange(0,dim-j+1):
        result += flag(k+j, k) * rho(k, us) * lkc_ambient[k+j] * mu**(k/2.)
    return result
    
def LKC_Chi2(j, us, mu, dof=2, dim=None, lkc_ambient=None):
    """Compute the expected value of the Lipschitz–Killing Curvatures (LKC) of the excursion set  for $\Chi^2_2$ fields.

    Parameters
    ----------
    j : int
        The index of the LKC.
        
    us : np.array
        The thresholds where the LKC is evaluated.

    mu : float
        The value of the derivative of the covariance function at the origin for the original Gaussian fields.

    dof : int, optional
        Number of degrees of freedom of the $\Chi^2$ field (i.e., number of squared Gaussian distributions that are summed). 
        Default: 2.
        
    dim : int, optional
        The dimension of the ambient manifold.
        
    lkc_ambient : np.array, optional
        An array of the Lipschitz–Killing Curvatures of the ambient manifold. Its lenght must be `dim+1` if `dim` is also given.
        
    Returns
    ----------
    LKC : np.array
        The expected value of the Lipschitz–Killing Curvatures at the thresholds.
    """
    dim, lkc_ambient = __prepare_lkc(dim, lkc_ambient)
    result = np.zeros_like(us)
    for k in np.arange(0,dim-j+1):
        result += flag(k+j, k) * rho_Chi2(k, dof, us) * lkc_ambient[k+j] * mu**(k/2.)
    return result
    

lkc_ambient_dict = {"2D":np.array([0., 0., 1.]),
                    "3D":np.array([0., 0., 0., 1.]),
                    "sphere":np.array([0., 0., 4.*np.pi]) / (4.*np.pi),
                    "SO3":np.array([0., 6.*np.pi, 0., 4.*np.pi**2]) / (4.*np.pi**2)}
"""Dictionary with the characterization of different spaces through their Lipschitz–Killing Curvatures"""


__all__ = ["get_μ", "define_mu", "flag", "omega", "rho", "rho_Chi2", "LKC", "LKC_P2", "lkc_ambient_dict"]

__docformat__ = "numpy"

 
