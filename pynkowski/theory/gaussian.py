"""This submodule contains the definition for general Isotropic Gaussian fields, as well as several classes of 
Isotropic Gaussian fields defined on particular spaces.
"""
import warnings
import numpy as np
from scipy.special import comb, gamma
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.integrate import quad
from .base_th import TheoryField
from .utils_th import get_μ, get_σ, LKC, lkc_ambient_dict, egoe, get_C2


class Gaussian(TheoryField):
    """General class for Isotropic Gaussian fields, to be used directly or as the base for specific Gaussian fields.

    Parameters
    ----------
    dim : int
        Dimension of the space where the field is defined.
    
    sigma : float, optional
        The standard deviation of the field.
        Default : 1.
        
    mu : float, optional
        The derivative of the covariance function at the origin, times $-2$. Equal to the variance of the first derivatives of the field.
        Default : 1.

    nu : float, optional
        The second derivative of the covariance function at the origin.
        Default : 1.
        
    lkc_ambient : np.array or None, optional
        The values for the Lipschitz–Killing Curvatures of the ambient space. If `None`, it is assumed that the volume is 1 and the rest is 0. This is exact for many spaces like Euclidean spaces or the sphere, and exact to leading order in μ for the rest.
        Default : None
        
    Attributes
    ----------
    dim : int
        Dimension of the space where the field is defined.
    
    name : str
        Name of the field, `"Isotropic Gaussian"` by default.
    
    sigma : float
        The standard deviation of the field.
        
    mu : float
        The derivative of the covariance function at the origin, times $-2$. Equal to the variance of the first derivatives of the field.

    nu : float
        The second derivative of the covariance function at the origin. Equal to $12$ times the variance of the second derivatives of the field ($4$ times for the cross derivative).
        
    lkc_ambient : np.array or None
        The values for the Lipschitz–Killing Curvatures of the ambient space.
        
    """   
    def __init__(self, dim, sigma=1., mu=1., nu=1., lkc_ambient=None):
        super().__init__(dim, name='Isotropic Gaussian', sigma=sigma, mu=mu, nu=nu, lkc_ambient=lkc_ambient)
        self._warn_non_euclidean = True
        
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
        return LKC(j, us, self.mu, dim=self.dim, lkc_ambient=self.lkc_ambient)/self.lkc_ambient[-1]
    
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
        return 1./comb(self.dim,j)*self.LKC(self.dim-j, us)
    
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
        return 1. - norm.cdf(us)
    
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

    def maxima_total(self):
        """Compute the expected values of local maxima of the field.

        Returns
        -------
        number_total : float
            Expected value of the number of local maxima.

        """
        if self._warn_non_euclidean:
            warnings.warn('If the ambient space is not euclidean, this function is an approximation. You can use `SphericalGaussian` or `EuclideanGaussian` instead.')
        rho1 = -0.5*self.mu
        return (2./np.pi)**((self.dim+1.)/2.) * gamma((self.dim+1.)/2.) * (-self.nu/rho1)**(self.dim/2.) * egoe(self.dim, 1., 0.) * self.lkc_ambient[-1]

    def maxima_dist(self, us):
        """Compute the expected distribution of local maxima of the field, as a function of threshold.

        Parameters
        ----------
        us : np.array
            The thresholds considered for the computation.

        Returns
        -------
        density : np.array
            Density distribution of the local maxima.

        """
        rho1 = -0.5*self.mu
        κ = -rho1 / np.sqrt(self.nu)
        assert κ <= (self.dim + 2.)/self.dim, '`0.5*mu/sqrt(nu)` must be $≤ (dim+2)/dim$'
        if self._warn_non_euclidean:
            warnings.warn('If the ambient space is not euclidean, this function is an approximation. You can use `SphericalGaussian` or `EuclideanGaussian` instead.')
        return np.real(np.sqrt(1./(1.-κ**2.+ 0.j)) * norm.pdf(us) * egoe(self.dim, 1./(1.-κ**2.), κ*us/np.sqrt(2.)) / egoe(self.dim, 1., 0.))

    def minima_total(self):
        """Compute the expected values of local minima of the field.

        Returns
        -------
        number_total : float
            Expected value of the number of local minima.

        """
        return self.maxima_total()

    def minima_dist(self, us):
        """Compute the expected distribution of local minima of the field, as a function of threshold.

        Parameters
        ----------
        us : np.array
            The thresholds considered for the computation.

        Returns
        -------
        density : np.array
            Density distribution of the local minima.

        """
        return self.maxima_dist(-us)
    
    
class SphericalGaussian(Gaussian):
    """Class for Spherical Isotropic Gaussian fields.

    Parameters
    ----------
    cls : np.array
        Angular power spectrum of the field.
    
    normalise : bool, optional
        If `True`, normalise the field to unit variance.
        Default : True

    fsky : float, optional
        Fraction of the sky covered by the field, `0<fsky<=1`.
        Default : 1.
    
    Attributes
    ----------
    cls : np.array
        Angular Power Spectrum of the field.
    
    fsky : float
        Fraction of the sky covered by the field.

    dim : int
        Dimension of the space where the field is defined, in this case this is 2.
    
    name : str
        Name of the field, `"Spherical Isotropic Gaussian"` by default.
    
    sigma : float
        The standard deviation of the field.

    mu : float
        The derivative of the covariance function at the origin, times $-2$. Equal to the variance of the first derivatives of the field.

    nu : float
        The second derivative of the covariance function at the origin. 
        
    C2 : float
        The second derivative of the angular covariance function at 1. 
        
    lkc_ambient : np.array or None
        The values for the Lipschitz–Killing Curvatures of the ambient space.
        
    """   
    def __init__(self, cls, normalise=True, fsky=1.):
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
        super().__init__(2, sigma=self.sigma, mu=self.mu, nu=self.nu, lkc_ambient=lkc_ambient_dict["sphere"]*self.fsky)
        self.name = 'Spherical Isotropic Gaussian'

    def maxima_total(self):
        """Compute the expected values of local maxima of the field.

        Returns
        -------
        number_total : float
            Expected value of the number of local maxima.

        """
        k1 = self.mu/self.C2
        return np.sqrt(2./(1.+k1)) / np.pi**((self.dim+1.)/2.) * k1**(-self.dim/2) * gamma((self.dim+1.)/2.) * egoe(self.dim,1./(1.+k1),0.) * self.lkc_ambient[-1]

    def maxima_dist(self, us):
        """Compute the expected distribution of local maxima of the field, as a function of threshold.

        Parameters
        ----------
        us : np.array
            The thresholds considered for the computation.

        Returns
        -------
        density : np.array
            Density distribution of the local maxima.

        """
        k1 = self.mu/self.C2
        k2 = self.mu**2./self.C2
        return np.real(np.sqrt((1.+k1)/(1.+k1-k2) +0.j) * norm.pdf(us) * egoe(self.dim, 1./(1.+k1-k2), np.sqrt(k2/2.)*us) / egoe(self.dim, 1./(1.+k1), 0.))

# class EuclideanGaussian(Gaussian):
#     """Class for Isotropic Gaussian fields defined in an Euclidean space (such as a 2D plane or a 3D cube). 

#     Parameters
#     ----------
#     dim : int
#         Dimension of the space where the field is defined.

#     Pk : np.array or function
#         Power spectrum of the field. It can be an array with the values of the power spectrum at wavenumbers `k`, or a function Pk(k) which returns the power spectrum at wavenumbers `k`. The latter is more accurate.

#     k : np.array
#         If `Pk` is an array, wavenumbers corresponding to the power spectrum. If `Pk` is a function, `k` is the array `[kmin, kmax]`.
    
#     normalise : bool, optional
#         If `True`, normalise the field to unit variance.
#         Default : True

#     volume : float, optional
#         Hausdorff measure of the space where the field is defined (area if `dim=2`, volume if `dim=3`, and so on).
#         Default : 1.
    
#     Attributes
#     ----------
#     dim : int
#         Dimension of the space where the field is defined.

#     Pk : function
#         Power spectrum of the field, to be evaluated at wavenumbers `k`.

#     klim : np.array
#         Minimum and maximum wavenumbers to be considered for the power spectrum `Pk`.
    
#     name : str
#         Name of the field, `"Euclidean Isotropic Gaussian"` by default.
    
#     sigma : float
#         The standard deviation of the field.

#     mu : float
#         The derivative of the covariance function at the origin, times $-2$. Equal to the variance of the first derivatives of the field.

#     nu : float
#         The second derivative of the covariance function at the origin. Equal to $12$ times the variance of the second derivatives of the field ($4$ times for the cross derivative).
        
#     lkc_ambient : np.array
#         The values for the Lipschitz–Killing Curvatures of the ambient space.
        
#     """   
#     def __init__(self, dim, Pk, k, normalise=True, volume=1.):
#         if type(Pk) is np.ndarray:
#             self.Pk = interp1d(k, Pk)
#             self.klim = np.array([k.min(), k.max()])
#         else:
#             assert callable(Pk), 'The power spectrum must be a function or an array'
#             assert k.shape == (2,), 'The power spectrum must be an array with two elements: `kmin, kmax`'
#             self.klim = k
#             self.Pk = Pk
        
#         if dim != 3:
#             warnings.warn('The formulae for the computation of sigma, mu, and nu are valid for `dim=3`. Please verify and possibly redefine these quantities manually.')
#             # Are these formulae valid also for other dimensions?
#         k2Pk = lambda k: self.Pk(k)*k**2.
#         sigma = np.sqrt(quad(k2Pk, self.klim[0], self.klim[1])[0] / (2.*np.pi**2.)) #* volume

#         k4Pk = lambda k: self.Pk(k)*k**4.
#         mu = quad(k4Pk, self.klim[0], self.klim[1])[0] / (6.*np.pi**2.) #* volume

#         k6Pk = lambda k: self.Pk(k)*k**6.
#         nu = quad(k6Pk, self.klim[0], self.klim[1])[0] / (120.*np.pi**2.) #* volume

#         if normalise:
#             mu = mu/sigma**2.
#             nu = nu/sigma**2.
#             self.Pk = lambda k: self.Pk(k)/sigma**2.
#             sigma = 1.
        
#         lkc_ambient = np.zeros(dim+1)
#         lkc_ambient[-1] = volume
#         super().__init__(dim=dim, sigma=sigma, mu=mu, nu=nu, lkc_ambient=lkc_ambient)
#         self.name = 'Euclidean Isotropic Gaussian'
#         self._warn_non_euclidean = False

#     def maxima_total(self):
#         """Compute the expected values of local maxima of the field.

#         Returns
#         -------
#         number_total : float
#             Expected value of the number of local maxima.

#         """
#         rho1 = -0.5*self.mu
#         return (2./np.pi)**((self.dim+1.)/2.) * gamma((self.dim+1.)/2.) * (-self.nu/rho1)**(self.dim/2.) * egoe(self.dim, 1., 0.) * self.lkc_ambient[-1]

#     def maxima_dist(self, us):
#         """Compute the expected distribution of local maxima of the field, as a function of threshold.

#         Parameters
#         ----------
#         us : np.array
#             The thresholds considered for the computation.

#         Returns
#         -------
#         density : np.array
#             Density distribution of the local maxima.

#         """
#         rho1 = -0.5*self.mu
#         κ = -rho1 / np.sqrt(self.nu)
#         assert κ <= (self.dim + 2.)/self.dim, '`0.5*mu/sqrt(nu)` must be $≤ (dim+2)/dim$'
#         return np.real(np.sqrt(1./(1.-κ**2.+ 0.j))) * norm.pdf(us) * egoe(self.dim, 1./(1.-κ**2.), κ*us/np.sqrt(2.)) / egoe(self.dim, 1., 0.)


    

__all__ = ["SphericalGaussian", "Gaussian"] #, "EuclideanGaussian", 

__docformat__ = "numpy"

 