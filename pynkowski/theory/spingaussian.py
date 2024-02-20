"""This submodule contains the definition for spin field with isotropic Gaussian Q and U.
"""
import numpy as np
from .utils_th import get_μ, get_σ, lkc_ambient_dict, get_C2, flag, rho
from .gaussian import Gaussian
 
from .base_th import _prepare_lkc


    
def LKC_spin(j, us, mu, dim=3, lkc_ambient=lkc_ambient_dict["SO3"], Ks=[1., 0.31912, 0.7088, 1.]):
    """Compute the expected value of the Lipschitz–Killing Curvatures (LKC) of the excursion set for Gaussian Isotropic fields.

    Parameters
    ----------
    j : int
        The index of the LKC.
        
    us : np.array
        The thresholds where the LKC is evaluated.

    mu : float
        The value of the derivative of the covariance function at the origin for the U and Q fields.
        
    dim : int, optional
        The dimension of the ambient manifold, SO(3). Must be 3.
        
    lkc_ambient : np.array, optional
        An array of the Lipschitz–Killing Curvatures of the ambient manifold. Its lenght must be `dim+1` if `dim` is also given.

    Ks : np.array, optional
        The normalisation factors for the Minkowski Functionals. 
        
    Returns
    ----------
    LKC : np.array
        The expected value of the Lipschitz–Killing Curvatures at the thresholds.
    """
    dim, lkc_ambient = _prepare_lkc(dim, lkc_ambient)
    assert dim==3, "The LKC for spin fields is only defined on SO(3), which has `dim=3`"
    result = np.zeros_like(us)
    KLs = Ks[::-1]              # To get the order as the LKC index, not the MF index
    KLs[-1] *= 2.**1.5 / 5.       # This is the determinant of the covariance matrix of the derivatives of the field
    KLs /= np.array([1., 1., mu**0.5, mu])
    for k in np.arange(0,dim-j+1):
        result += flag(k+j, k) * rho(k, us) * lkc_ambient[k+j] / KLs[k+j]
    return result * KLs[j] /lkc_ambient[-1]




class SpinGaussian(Gaussian):
    """Class for Spin Isotropic Gaussian fields in the SO(3) formalism.

    Parameters
    ----------
    cls : np.array
        Angular power spectrum of the field (cls_E + cls_B).
    
    normalise : bool, optional
        If `True`, normalise the field to unit variance.
        Default : True

    fsky : float, optional
        Fraction of the sky covered by the field, `0<fsky<=1`.
        Default : 1.

    Ks : np.array, optional
        The normalisation constants for the MFs of the field: [K_0, K_1, K_2, K_3].
        Default : [1., 0.31912, 0.7088, 1.] as found in Carrón Duque et al (2023)

    leading_order : bool, optional
        Whether to use only the leading order in μ for the computation of the MFs or the exact expression (with two terms).
        Default : False (exact expression)
    
    Attributes
    ----------
    cls : np.array
        Angular Power Spectrum of the field (cls_E + cls_B).
    
    fsky : float
        Fraction of the sky covered by the field.

    dim : int
        Dimension of the space where the field is defined, in this case this is 3.
    
    name : str
        Name of the field, `"Spin Isotropic Gaussian"` by default.
    
    sigma : float
        The standard deviation of the field.

    mu : float
        The derivative of the covariance function at the origin, times $-2$ (in the spatial coordinates θϕ). Equal to the variance of the first derivatives of the field.

    nu : float
        The second derivative of the covariance function at the origin (in the spatial coordinates θϕ). 
        
    C2 : float
        The second derivative of the angular covariance function at 1 (in the spatial coordinates θϕ). 
        
    lkc_ambient : np.array or None
        The values for the Lipschitz–Killing Curvatures of the ambient space.

    Ks : np.array
        The normalisation constants for the MFs of the field: [K_0, K_1, K_2, K_3].
        
    leading_order : bool
        Whether to use only the leading order in μ for the computation of the MFs or the exact expression (with two terms).

    """   
    def __init__(self, cls, normalise=True, fsky=1., Ks=[1., 0.31912, 0.7088, 1.], leading_order=True):
        if normalise:
            cls /= get_σ(cls)
            self.sigma = 1.
        else:
            self.sigma = np.sqrt(get_σ(cls))
        self.cls = cls
        self.fsky = fsky
        self.mu = get_μ(cls)
        self.C2 = get_C2(cls)
        self.nu = self.C2/4. - self.mu/24.
        super().__init__(3, sigma=self.sigma, mu=self.mu, nu=self.nu, lkc_ambient=lkc_ambient_dict["SO3"]*self.fsky)
        self.leading_order = leading_order
        if leading_order:
            self.lkc_ambient[1] = 0.
        self.name = 'Spin Isotropic Gaussian'
        self.Ks = Ks

    def LKC(self, j, us):
        """Compute the expected values of the Lipschitz–Killing Curvatures of the excursion sets at thresholds `us`, $\mathbb{L}_j(A_u(f))$.

        Parameters
        ----------
        j : int
            Index of the LKC to compute, `0 < j < dim`.
            
        us : np.array
            The thresholds considered for the computation.

        Returns
        -------
        lkc : np.array
            Expected value of LKC evaluated at the thresholds.

        """
        us /= self.sigma
        return LKC_spin(j, us, self.mu, dim=self.dim, lkc_ambient=self.lkc_ambient, Ks=self.Ks)


__all__ = ["SpinGaussian"]