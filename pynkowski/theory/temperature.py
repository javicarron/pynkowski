import numpy as np
import scipy.stats

from .utils import get_μ, define_mu, define_us_for_V

norm = scipy.stats.norm()



def V0_th_s0(u):
    """Compute the expected value of the normalised first MF v0 at threshold u for a Gaussian isotropic scalar field normalised for its standard deviation.

    Parameters
    ----------
    u : np.array
        Thresholds at which the first MF is computed.
    
    Returns
    -------
    v0 : np.array
        The expected value of the normalised first MF at thresholds u.
    
    """    
    return (1-norm.cdf(u))

def V1_th_s0(u, μ):
    """Compute the expected value of the normalised second MF v1 at threshold u for a Gaussian isotropic scalar field normalised for its standard deviation.

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
    return np.sqrt(μ) * np.exp(-u**2./2.) / 8.

def V2_th_s0(u, μ):
    """Compute the expected value of the normalised third MF v2 at threshold u for a Gaussian isotropic scalar field normalised for its standard deviation.

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
    return μ * u * np.exp(-u**2. /2.) / np.sqrt(2.*np.pi)**3. 



class TheoryTemperature():
    """Class to compute the expected values of Minkowski functionals (MFs) for a Gaussian isotropic scalar field normalised for its standard deviation defined on the sphere 
    like the temperature anisotropies of the CMB.

    Parameters
    ----------
    us : np.array, optional
        The thresholds at which the theoretical MFs will be computed. 
        If not given, a range between -5σ and 5σ with steps of 0.1σ is considered, 
        with σ=1 the expected standard deviation of the field.
    
    cls : np.array, optional
        The angular power spectrum of the Gaussian isotropic scalar field. 
        Default : None 
    
    μ : float, optional
        The derivative of the covariance function at the origin for the Gaussian isotropic scalar field.
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
        The derivative of the covariance function at the origin for the Gaussian isotropic scalar field.
        
    """    
    def __init__(self, us=None, cls=None, μ=None, average_bin=True, edges=False):
        """Class to compute the expected values of Minkowski functionals (MFs) for a Gaussian isotropic scalar field defined on the sphere.

        """    
        if (us is None):
            Δu = 0.1
            self.us = np.arange(-5+Δu/2., 5.+Δu/2., Δu)
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

             
        self.μ = define_mu(cls,μ)   
        if not average_bin:
            self.dus = 0.*self.dus
        
    
    def V0(self):
        """Compute the expected values of the normalised first MF v0 at the different thresholds us.
        $$\mathbb{E}[ v_{0} ] =1 -\Phi (u)$$
        where $\Phi$ is the cumulative normal distribution.
        """    
        return (V0_th_s0(self.us))

    def V1(self):
        """Compute the expected values of the normalised second MF v1 at the different thresholds us.
        
        $$\mathbb{E}[ v_{1}(u) ] = {1 \over 8} \exp{(- {u^2 \over 2})} \mu^{1/2}$$
        """ 
        us_ = define_us_for_V(self.us,self.dus)
        v1_ = V1_th_s0(us_, self.μ)
        
        return np.mean(v1_, axis=1)

    def V2(self):
        """Compute the expected values of the normalised third MF v2 at the different thresholds us.
        
        $$\mathbb{E}[v_{2}(u)] = {2 \over \sqrt{(2\pi)^{3}}} \exp(-{u^{2} \over 2}) u$$
        """    
        us_ = define_us_for_V(self.us,self.dus)
        v2_ = V2_th_s0(us_, self.μ)
        
        return np.mean(v2_, axis=1)



__all__ = ["TheoryTemperature"]

__docformat__ = "numpy"

