import numpy as np
import scipy.stats

norm = scipy.stats.norm()


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

def V0_th_s2(u):
    """Compute the expected value of the normalised first MF v0 at threshold u for a Gaussian field defined in SO(3) normalised for its standard deviation.

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

def V1_th_s2(u, μ):
    """Compute the expected value of the normalised second MF v1 at thresholds u for a Gaussian field defined in SO(3) normalised for its standard deviation.

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
    return 2.*np.sqrt(2*np.pi**3 *μ) * norm.pdf(u) / (4*np.pi**2)
    
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
    
def V2_th_s2(u, μ):
    """Compute the expected value of the normalised third MF v2 at threshold u for a Gaussian field defined in SO(3) normalised for its standard deviation.

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
    return (((1-norm.cdf(u)) * 3.*np.pi + u*norm.pdf(u) * μ * 4.*np.pi) / (4*np.pi**2)) / (6.*np.pi)   #This has a ratio of ~0.13

def V3_th_s2(u, μ):
    """Compute the expected value of the normalised fourth MF v3 at threshold u for a Gaussian field defined in SO(3) normalised for its standard deviation.

    Parameters
    ----------
    u : np.array
        Threshold at which the fourth MF is computed.
    
    μ : float
        The derivative of the covariance function at the origin for the Gaussian isotropic scalar field.
    
    Returns
    -------
    v3 : np.array
        The expected value of the normalised fourth MF at threshold u.
    
    """    
    return μ*np.exp(-u**2. /2.) * (u**2. -1.) / 8
    
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
    
    μ : float, scalar, optional
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
        
    μ : float, scalar
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

        """    
        return (V0_th_s0(self.us))

    def V1(self):
        """Compute the expected values of the normalised second MF v1 at the different thresholds us.

        """ 
        us_ = define_us_for_V(self.us,self.dus)
        v1_ = V1_th_s0(us_, self.μ)
        
        return np.mean(v1_, axis=1)

    def V2(self):
        """Compute the expected values of the normalised third MF v2 at the different thresholds us.

        """    
        us_ = define_us_for_V(self.us,self.dus)
        v2_ = V2_th_s0(us_, self.μ)
        
        return np.mean(v2_, axis=1)



class TheoryP2():
    """Class to compute the expected values of Minkowski functionals (MFs) for the sum of two squared Gaussian isotropic fields normalised for their standard deviations defined on the sphere 
    like the polarised intensity of the CMB (P^2 = Q^2 + U^2).

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
    
    μ : float, scalar, optional
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
        
    μ : float, scalar
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

        """    
        return (V0_th_P2(self.us))

    def V1(self):
        """Compute the expected values of the normalised second MF v1 at the different thresholds us.

        """ 
        us_ = define_us_for_V(self.us,self.dus)
        v1_ = V1_th_P2(us_, self.μ)
        
        return np.mean(v1_, axis=1)
    
    def V2(self):
        """Compute the expected values of the normalised third MF v2 at the different thresholds us.

        """    
        us_ = define_us_for_V(self.us,self.dus)
        v2_ = V2_th_P2(us_, self.μ)
        
        return np.mean(v2_, axis=1)




__all__ = ["TheoryP2",
           "TheoryTemperature",
           "get_μ"]
