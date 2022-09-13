import numpy as np
import scipy.stats

from .utils import get_μ, define_mu, define_us_for_V

norm = scipy.stats.norm()



def V0_th_P2(u):
    """Compute the expected value of the normalised first MF v0 at threshold u for the sum of two squared Gaussian isotropic fields normalised for their standard deviations.

    Parameters
    ----------
    u : np.array
        Thresholds at which the first MF is computed.
    
    Returns
    -------
    v0 : np.array
        The expected value of the normalised first MF at thresholds u.
    
    """    
    return 1-((1-np.exp(-u/2.)) * (u>=0.))

def V1_th_P2(u, μ):
    """Compute the expected value of the normalised second MF v1 at threshold u for the sum of two squared Gaussian isotropic fields normalised for their standard deviations.

    Parameters
    ----------
    u : np.array
        Thresholds at which the second MF is computed.
    
    μ : float
        The derivative of the covariance function at the origin for the Gaussian isotropic scalar field.
    
    Returns
    -------
    v1 : np.array
        The expected value of the normalised second MF at thresholds u.
    
    """    
    return np.sqrt(μ * (u) / 4.) * np.exp(-u/2.) * (u>=0.) * (np.sqrt(np.pi/8.))

def V2_th_P2(u, μ):
    """Compute the expected value of the normalised third MF v2 at threshold u for the sum of two squared Gaussian isotropic fields normalised for their standard deviations.

    Parameters
    ----------
    u : np.array
        Thresholds at which the third MF is computed.
    
    μ : float
        The derivative of the covariance function at the origin for the Gaussian isotropic scalar field.
    
    Returns
    -------
    v2 : np.array
        The expected value of the normalised third MF at thresholds u.
    
    """    
    return (((μ * (u-1.) * np.exp(-u/2.)) / (2.*np.pi) ) * (u>=0.)) 





class TheoryP2():
    """Class to compute the expected values of Minkowski functionals (MFs) for the sum of two squared Gaussian isotropic fields normalised for their standard deviations defined on the sphere 
    like the polarised intensity of the CMB ($P^2 = Q^2 + U^2$).

    Parameters
    ----------
    us : np.array, optional
        The thresholds at which the theoretical MFs will be computed. 
        If not given, a range between 0 and 5σ with steps of 0.1σ is considered, 
        with σ=1 the expected standard deviation of the fields U and Q.
    
    cls : np.array, optional
        The angular power spectrum associated to the Gaussian isotropic fields. 
        Shape '(..., lmax+1)'. '...' can be 2 (EE, BB) or absent (assumed to be EE+BB).
        Default : None 
    
    μ : float, optional
        The derivative of the covariance function at the origin for each of the two independent Gaussian isotropic fields (i.e., U and Q in the cosmological case).
        If both μ and cls are given, an error will be raised.
        If only cls is given, μ will be computed from input cls.
        If neither μ nor cls are given, μ is assumed to be 1.
        Default : None
        
    average_bin : bool, optional
        If True, the results of V1 and V2 are the average on each bin, to be compared with binned computations on maps.
        If False, the results are the evaluation on the center of each bin.
        The value is always exactly computed for V0, as the computation on maps does not imply binning.
        Defaul : True
        
    edges : bool, optional
        If False, the given 'us' is considered as an array of uniformly distributed thresholds. 
        If True, input 'us' is considered as a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform distributions of thresholds. 
        In this last case, the thresholds are the central value of the given bins.
        Neglected if 'us' is not given.
        Default : False.

    Attributes
    ----------
    us : np.array
        The thresholds at which the theoretical MFs are computed. 
        
    μ : float
        The derivative of the covariance function at the origin for the sum of two squared Gaussian isotropic fields.
        
    """    
    def __init__(self, us=None, Cls=None, μ=None, average_bin=True, edges=False):
        """Class to compute the expected values of Minkowski functionals (MFs) for a Gaussian isotropic scalar field defined on the sphere.

        """    
        if (us is None):
            Δu = 0.05
            self.us = np.arange(Δu/2., 5.+Δu/2., Δu)
            self.dus = Δu*np.ones(self.us.shape[0])
        else:
            us = np.array(us)
            if us.shape == (1,):
                self.us = us
                self.dus = 0.
            else:
                if edges:
                    self.dus = (us[1:]-us[:-1])
                    self.us = (us[1:]+us[:-1])/2.
                else:
                    self.us = us
                    self.dus = (us[1]-us[0])*np.ones(us.shape[0])

        if (Cls is not None) and (μ is None):
            if (Cls.ndim == 2) and (Cls.shape[0]==2):
                cls = (cls[0]+cls[1])/2.
            elif (Cls.ndim == 1):
                cls = cls/2.
            else:
                raise ValueError(r"Cls dimension has to be either (2,lmax+1) or (lmax+1)")
                
        self.μ = define_mu(Cls,μ)   
        if not average_bin:
            self.dus = 0.*self.dus
        
    
    def V0(self):
        """Compute the expected values of the normalised first MF v0 at the different thresholds us.

        $$\mathbb{E}[{v_{0}(u)}] = \exp (-u/2)$$
        """    
        return (V0_th_P2(self.us))

    def V1(self):
        """Compute the expected values of the normalised second MF v1 at the different thresholds us.
        
        $$\mathbb{E}[{v_{1}(u)}] = {\sqrt{2\pi } \over 8} \sqrt{\mu u}\exp (-{u \over 2})$$
        """ 
        us_ = define_us_for_V(self.us,self.dus)
        v1_ = V1_th_P2(us_, self.μ)
        
        return np.mean(v1_, axis=1)
    
    def V2(self):
        """Compute the expected values of the normalised third MF v2 at the different thresholds us.
        
        $$\mathbb{E}[{v_{2}(u)}] = \mu {(u-1)\exp (-u/2) \over 2\pi }$$
        """    
        us_ = define_us_for_V(self.us,self.dus)
        v2_ = V2_th_P2(us_, self.μ)
        
        return np.mean(v2_, axis=1)


__all__ = ["TheoryP2"]

__docformat__ = "numpy"

