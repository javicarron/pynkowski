'''This submodule contains the classes for the theoretical $\chi^2$ fields.'''
import numpy as np
from scipy.special import comb
from .utils_th import LKC_Chi2, get_σ, get_μ, lkc_ambient_dict
from .base_th import TheoryField
        
class Chi2(TheoryField):
    """General class for Isotropic Chi² fields, to be used directly or as the base for specific Chi² fields.

    Parameters
    ----------
    dim : int
        Dimension of the space where the field is defined.
        
    dof : int, optional
        Number of degrees of freedom $dof$ of the field: $\Chi^2 = \sum_{i=0}{dof} N_i^2$.
        Default: 2
    
    sigma : float, optional
        The standard deviation of the constituent Gaussian fields $N_i$, such that $\Chi^2 = \sum_{i=0}{dof} N_i^2$.
        Default: 1.
        
    mu : float, optional
        The derivative of the covariance function at the origin for the constituent Gaussian fields.
        Default: 1.

    nu : float, optional
        The second derivative of the covariance function at the origin for the constituent Gaussian fields.
        Default: 1.
        
    lkc_ambient : np.array or None, optional
        The values for the Lipschitz–Killing Curvatures of the ambient space. If `None`, it is assumed that the volume is 1 and the rest is 0. This is exact for many spaces like Euclidean spaces or the sphere, and exact to leading order in μ for the rest.
        Default: None
        
    Attributes
    ----------
    dim : int
        Dimension of the space where the field is defined.
    
    name : str
        Name of the field, `"Isotropic Chi²"` by default.

    dof : int, optional
        Number of degrees of freedom $dof$ of the field: $\Chi^2 = \sum_{i=0}{dof} N_i^2$.
    
    sigma : float
        The standard deviation of the field.
        
    mu : float
        The derivative of the covariance function at the origin.

    nu : float, optional
        The second derivative of the covariance function at the origin for the constituent Gaussian fields.
        
    lkc_ambient : np.array or None
        The values for the Lipschitz–Killing Curvatures of the ambient space.
        
    """   
    def __init__(self, dim, dof=2, sigma=1., mu=1., nu=1., lkc_ambient=None):
        super().__init__(dim, name='Isotropic Chi²', sigma=sigma, mu=mu, nu=nu, lkc_ambient=lkc_ambient)
        self.dof = dof
        
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
        return LKC_Chi2(j, us, self.mu, dof=self.dof, dim=self.dim, lkc_ambient=self.lkc_ambient)/self.lkc_ambient[-1]
    
    def MF(self, j, us):
        """Compute the expected values of the Minkowski Functionals of the excursion sets at thresholds `us`, $V_j(A_u(f))$.

        Parameters
        ----------
        j : int
            Index of the MF to compute, `0 < j < dim`.
            
        us : np.array
            The thresholds considered for the computation.

        Returns
        -------
        mf : np.array
            Expected value of MFs evaluated at the thresholds.

        """
        us /= self.sigma
        return 1./comb(self.dim,j)*LKC_Chi2(self.dim-j, us, self.mu, dof=self.dof, dim=self.dim, lkc_ambient=self.lkc_ambient)/self.lkc_ambient[-1]
    
    def V0(self, us):
        """Compute the expected values of the first Minkowski Functionals of the excursion sets at thresholds `us`, $V_0(A_u(f))$.

        Parameters
        ----------
        us : np.array
            The thresholds considered for the computation.

        Returns
        -------
        v0 : np.array
            Expected value of $V_0$ evaluated at the thresholds.

        """
        us /= self.sigma
        return self.MF(0, us)
    
    def V1(self, us):
        """Compute the expected values of the first Minkowski Functionals of the excursion sets at thresholds `us`, $V_1(A_u(f))$.

        Parameters
        ----------
        us : np.array
            The thresholds considered for the computation.

        Returns
        -------
        v1 : np.array
            Expected value of $V_1$ evaluated at the thresholds.

        """
        us /= self.sigma
        return self.MF(1, us)
    
    def V2(self, us):
        """Compute the expected values of the first Minkowski Functionals of the excursion sets at thresholds `us`, $V_2(A_u(f))$. Only for fields with `dim ≥ 2`.

        Parameters
        ----------
        us : np.array
            The thresholds considered for the computation.

        Returns
        -------
        v2 : np.array
            Expected value of $V_2$ evaluated at the thresholds.

        """
        assert self.dim>=2, 'V2 is defined only for fields with dim≥2'
        us /= self.sigma
        return self.MF(2, us)
    
    def V3(self, us):
        """Compute the expected values of the first Minkowski Functionals of the excursion sets at thresholds `us`, $V_3(A_u(f))$. Only for fields with `dim ≥ 3`.

        Parameters
        ----------
        us : np.array
            The thresholds considered for the computation.

        Returns
        -------
        v3 : np.array
            Expected value of $V_3$ evaluated at the thresholds.

        """
        assert self.dim>=3, 'V3 is defined only for fields with dim≥3'
        us /= self.sigma
        return self.MF(3, us)
    
    # def V4(self, us):
    #     assert self.dim>=4, 'V4 is defined only for fields with dim≥4'
    #     us /= self.sigma
    #     return self.MF(4, us)
    
    
class SphericalChi2(Chi2):
    """Class for Spherical Isotropic Gaussian fields, to be used directly or as the base for specific Gaussian fields.

    Parameters
    ----------
    cls : np.array
        Angular power spectrum of the polarization field (EE, BB). The shape is `(2,lmax+1)`
    
    normalise : bool, optional
        If `True`, normalise the Q and U fields to unit variance
        Default : True
    
    Attributes
    ----------
    cls : np.array
        Angular Power Spectrum of the polarization field.
    
    dim : int
        Dimension of the space where the field is defined, in this case this is 2.
    
    name : str
        Name of the field, `"Spherical Isotropic Chi²"` by default.

    dof : int, optional
        Number of degrees of freedom $dof$ of the field, $2$.
    
    sigma : float
        The standard deviation of the field.
        
    mu : float
        The derivative of the covariance function at the origin.

    nu : float, optional
        The second derivative of the covariance function at the origin for the constituent Gaussian fields.
        
    lkc_ambient : np.array or None
        The values for the Lipschitz–Killing Curvatures of the ambient space.
        
    """   
    def __init__(self, cls, normalise=True):
        if normalise:
            cls /= get_σ((cls[0]+cls[1])/2.)
            self.sigma = 1.
        else:
            self.sigma = np.sqrt(get_σ(cls))
        self.cls = cls
        self.mu = get_μ(cls[0]+cls[1])
        super().__init__(dim=2, dof=2, sigma=self.sigma, mu=self.mu, lkc_ambient=lkc_ambient_dict["sphere"])
        self.name = 'Spherical Isotropic Chi²'        
        


__all__ = ["SphericalChi2", "Chi2"]

__docformat__ = "numpy"

 