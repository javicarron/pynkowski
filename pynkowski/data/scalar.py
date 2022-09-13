import numpy as np
import healpy as hp

try:
    from tqdm.notebook import tqdm
except:
    tqdm = lambda x: x
    print('tqdm not loaded')
    
from .utils import get_theta, derivatives, second_derivatives
 
 
 

class Scalar():
    """Class to compute Minkowski functionals (MFs) and extrema of Healpix scalar maps. It computes and stores spatial first and
    second derivatives of Healpix scalar maps.

    Parameters
    ----------
    Smap : np.array
        The input Healpix map where all the statistics will be computed.
    
    normalise : bool, optional
        If True, divide input Smap by its standard deviation. Default: True.
    
    mask : np.array, optional
        An input Healpix mask. All the statiscal quantities will be computed only within this mask. Default: None (all sky is considered).
    
    Attributes
    ----------
    Smap : np.array
        The Healpix map where all the statistics are computed.
        
    nside : int
        The Nside parameter of Smap
    
    mask : np.array
        A Healpix mask whithin which all the statiscal quantities are computed.
        
    grad_phi : np.array
        The phi covariant first derivative of Smap (only if first or second MFs are computed).

    grad_theta : np.array
        The theta covariant first derivative of Smap (only if first or second MFs are computed).

    der_phi : np.array
        The phi partial first derivative of Smap (only if first or second MFs are computed). 

    der_theta_phi : np.array
        The second covariant derivative wrt theta and phi of Smap (only if second MF is computed). 

    der_phi_phi : np.array
        The second covariant derivative wrt phi of Smap (only if second MF is computed). 

    der_theta_theta : np.array
        The second covariant derivative wrt theta of Smap (only if second MF is computed). 

    Notes
    -----
    The derivatives are always computed full-sky regardless of input mask.
        
    """    
    def __init__(self, Smap, normalise=True, mask=None):
        """Initialise the class to compute Minkowski functionals (MFs) and extrema of Healpix scalar maps. 

        Parameters
        ----------
        Smap : np.array
            The input Healpix map where all the statistics will be computed.

        normalise : bool, optional
            If True, divide input Smap by its standard deviation. Default: True.

        mask : np.array, optional
            An input Healpix mask. All the statiscal quantities will be computed only within this mask. Default: None (all sky is considered).


        """    
        self.Smap = Smap.copy()
        self.nside = hp.get_nside(Smap)
        if mask is None:
            self.mask = np.ones(Smap.shape, dtype='bool')
        else:
            self.mask = mask
            if hp.get_nside(mask) != self.nside:
                raise ValueError('The map and the mask have different nside')
        if normalise:
            σ = self.get_variance()
            self.Smap /= np.sqrt(σ)
        self.grad_phi = None
        self.der_phi_phi = None
    
    
    def __repr__(self):
        return(f'map = {self.Smap}')
    
    def get_variance(self):
        """compute the variance of the input Healpix scalar map within the input mask. 

        Returns
        -------
        var : float
            The variance of the input Healpix map within the input mask.

        """    
        return (np.var(self.Smap[self.mask]))

    def set_pix(self, pixs):
        """return the values of the input Healpix scalar map in pixels pixs. 

        Parameters
        ----------
        pixs : np.array
            The indices of the input map pixels whose values of the map are returned.

        Returns
        -------
        values : np.array
            The values of the input Healpix scalar map in pixels pixs.

        """    
        return self.Smap[pixs]

    def get_gradient(self):
        """Compute the covariant and partial first derivatives of the input Healpix scalar map. 
        It stores:
        
        - first covariant derivative wrt theta in self.grad_theta
        - first partial derivative wrt phi in self.der_phi
        - first covariant derivative wrt phi in self.grad_phi

        """    
        S_grad = derivatives(self.Smap, gradient=True)
        theta = get_theta(self.nside)
        self.grad_theta = Scalar(S_grad[0], normalise=False)
        self.der_phi = Scalar(np.cos(theta) * S_grad[1], normalise=False)
        self.grad_phi = Scalar(S_grad[1], normalise=False)   
        
        
    def get_hessian(self):
        """compute the covariant second derivatives of the input Healpix scalar map. 
        It stores:
        
        - second covariant derivative wrt theta in self.der_theta_theta
        - second covariant derivative wrt phi in self.der_phi_phi
        - second covariant derivative wrt theta and phi in self.der_theta_phi

        """    
        if self.grad_phi == None:
            self.get_gradient()
        theta = get_theta(self.nside)
        
        S_der_der = second_derivatives(self.grad_theta.Smap, self.der_phi.Smap)
        
        self.der_theta_theta = Scalar(S_der_der[0], normalise=False)
        self.der_phi_phi = Scalar(S_der_der[1]/np.cos(theta)**2. + self.grad_theta.Smap*np.tan(theta), normalise=False)
        self.der_theta_phi = Scalar((S_der_der[2]/np.cos(theta) - self.grad_phi.Smap * np.tan(theta)) , normalise=False)
        

    def get_κ(self, pixs):
        """Compute the geodesic curvature multiplied by the modulus of the gradient in pixels pixs. If not already computed, it computes 
        the first and second covariant derivatives of the input map.

        Parameters
        ----------
        pixs : np.array
            The indices of the input map pixels where geodesic curvature is computed.

        Returns
        -------
        k : np.array
            The geodesic curvature in pixels `pixs`.

        """    
        num = 2.*self.grad_theta.set_pix(pixs)*self.grad_phi.set_pix(pixs)*self.der_theta_phi.set_pix(pixs) - self.grad_phi.set_pix(pixs)**2. * self.der_theta_theta.set_pix(pixs) - self.grad_theta.set_pix(pixs)**2. * self.der_phi_phi.set_pix(pixs)
        den = self.grad_theta.set_pix(pixs)**2. + self.grad_phi.set_pix(pixs)**2.
        return num / den
        

    def V0_pixel(self, u):
        """Determine where input Healpix scalar map Smap is greater than threshold u. 

        Parameters
        ----------
        u : float
            The threshold considered for the computation of first Minkowski functional V0.

        Returns
        -------
        v0map : np.array
            a bool array with the same shape as the input map, with False where input map values are lower than threshold u.

        """    
        return self.Smap>u
    
    def V0_iter(self, u):
        """Compute the normalised first Minkowski functional v0 at the threshold u within the given mask. 

        Parameters
        ----------
        u : float
            The threshold considered for the computation of v0.

        Returns
        -------
        v0 : np.array
            First normalised Minkowski functional evaluated at threshold u within the given mask.

        """    
        return np.mean(self.V0_pixel(u)[self.mask])
    
    def V0(self, us):
        """Compute the normalised first Minkowski functional v0 at the different thresholds us within the given mask.

        Parameters
        ----------
        us : np.array
            The thresholds considered for the computation of v0.

        Returns
        -------
        v0s : np.array
            First normalised Minkowski functional evaluated at thresholds us within the given mask.

        """    
        us = np.atleast_1d(us)
        return np.array([self.V0_iter(u) for u in tqdm(us)])
    
    
    def V1_pixel(self, u, du):
        """Compute the modulus of the gradient where the values of the input map are between `u-du/2` and `u+du/2`. 

        Parameters
        ----------
        u : float
            The centered value of the bin considered for the computation.

        du : float
            The width of the bin considered for the computation.

        Returns
        -------
        v1map : np.array
            Modulus of the gradient where `u-du/2 < Smap < u+du/2`.

        """    
        theta = get_theta(self.nside)
        
        areas = np.zeros_like(theta)

        mask = (u + du/2. > self.Smap) & (u - du/2. <= self.Smap) & ~(np.isclose(np.cos(theta),0, atol=1.e-2))
        pixs = np.arange(12*self.nside**2)[mask]
        
        areas[pixs] = np.sqrt(self.grad_theta.set_pix(pixs)**2. + self.grad_phi.set_pix(pixs)**2.)
        return areas/du / 4.
    
    def V1_iter(self, u, du):
        """Compute the normalised second Minkowski functional v1 in the bin `u-du/2` and `u+du/2` within the given mask. 

        Parameters
        ----------
        u : float
            The centered value of the bin considered for the computation of v1.

        du : float
            The width of the bin considered for the computation of v1.

        Returns
        -------
        v1 : float
            Second normalised Minkowski functional evaluated in the bin u-du/2 and u+du/2 within the given mask.

        """    
        return np.mean(self.V1_pixel(u, du)[self.mask])
    
    def V1(self, us, edges=False):
        """Compute the normalised second Minkowski functional v1 at the different thresholds us within the given mask.

        Parameters
        ----------
        us : np.array
            The thresholds considered for the computation of v2. See 'edges' for details.

        edges : bool, optional
            If False, us is considered as an array of uniformly distributed thresholds. 
            If True, us is considered as a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform distributions of thresholds. 
            In this last case, the thresholds are the central value of the given bins.
            Default: False.

        Returns
        -------
        v1s : np.array
            Second normalised Minkowski functional evaluated at thresholds us within the given mask.

        """
        if self.grad_phi == None:
            self.get_gradient()
            
        us = np.atleast_1d(us)
        
        if edges:
            dus = (us[1:]-us[:-1])
            us = (us[1:]+us[:-1])/2.
        else:
            if us.shape == (1,):
                dus = np.array([0.1])
            else:
                dus = (us[1]-us[0])*np.ones(us.shape[0])           
                if not (np.isclose(us[1:]-us[:-1], dus[0])).all():
                    raise ValueError('The thresholds distributions is not uniform. Please set edges=True')
            
        return np.array([self.V1_iter(u, du) for (u, du) in zip(tqdm(us), dus)])

    
    def V2_pixel(self, u, du):
        """Compute the geodesic curvature multiplied by the modulus of the gradient where the values of the input map are between `u-du/2` and `u+du/2`. 

        Parameters
        ----------
        u : float
            The centered value of the bin considered for the computation.

        du : float
            The width of the bin considered for the computation.

        Returns
        -------
        v2map : np.array
            Geodesic curvature multiplied by the modulus of the gradient where `u-du/2 < Smap < u+du/2`.

        """           
        theta = get_theta(self.nside)
        
        mask = (u + du/2. > self.Smap) & (u -du/2. <= self.Smap) & ~(np.isclose(np.cos(theta),0, atol=1.e-2))
        pixs = np.arange(12*self.nside**2)[mask]
        areas = np.zeros_like(theta)
        areas[mask] = self.get_κ(pixs)
        
        return (areas/du / (2.*np.pi))
    
     
    def V2_iter(self, u, du):
        """Compute the normalised third Minkowski functional v2 in the bin `u-du/2` and `u+du/2` within the given mask. 

        Parameters
        ----------
        u : float
            The centered value of the bin considered for the computation of v2.

        du : float
            The width of the bin considered for the computation of v2.

        Returns
        -------
        v2 : np.array
            Third normalised Minkowski functional evaluated in the bin u-du/2 and u+du/2 within the given mask.

        """    
        return np.mean(self.V2_pixel(u, du)[self.mask])
    
    def V2(self, us, edges=False):
        """Compute the normalised third Minkowski functional v2 at the different thresholds us within the given mask.

        Parameters
        ----------
        us : np.array
            The thresholds considered for the computation of v2. See 'edges' for details.

        edges : bool, optional
            If False, us is considered as an array of uniformly distributed thresholds. 
            If True, us is considered as a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform distributions of thresholds. 
            In this last case, the thresholds are the central value of the given bins.
            Default: False.

        Returns
        -------
        v2s : np.array
            Third normalised Minkowski functional evaluated at thresholds us within the given mask.

        """   
        if self.grad_phi == None:
            self.get_gradient()
        if self.der_phi_phi == None:
            self.get_hessian()
            
        us = np.atleast_1d(us)
        
        if edges:
            dus = (us[1:]-us[:-1])
            us = (us[1:]+us[:-1])/2.
        else:
            if us.shape == (1,):
                dus = np.array([0.1])
            else:
                dus = (us[1]-us[0])*np.ones(us.shape[0])           
                if not (np.isclose(us[1:]-us[:-1], dus[0])).all():
                    raise ValueError('The thresholds distributions is not uniform. Please set edges=True')
            
        return np.array([self.V2_iter(u, du) for (u, du) in zip(tqdm(us), dus)])
    

    
    def get_maxima(self):
        """Find the local maxima of the input scalar map.

        Returns
        -------
        pixels : np.array
            Indices of the pixels which are local maxima.

        values : np.array
            Values of input map which are local maxima.

        """    
        neigh = hp.get_all_neighbours(self.nside, np.arange(12*self.nside**2))
        
        extT = np.concatenate([self.Smap, [np.min(self.Smap)-1.]])
        neigh_matrix = extT[neigh]

        mask = np.all(self.Smap > neigh_matrix, axis=0)
        pixels = np.argwhere(mask).flatten()
        values = self.Smap[pixels].flatten()

        return(pixels, values)
    
    def get_minima(self):
        """Find the local minima of the input scalar map.

        Returns
        -------
        pixels : np.array
            Indices of the pixels which are local minima

        values : np.array
            Values of input map which are local minima

        """    
        self.Smap = -self.Smap
        pixels, values = self.get_maxima()
        self.Smap = -self.Smap
        
        return(pixels, -values)
        
        
        
__all__ = ["Scalar"]
__docformat__ = "numpy"
