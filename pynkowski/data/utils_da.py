'''This submodule contains some utilities for data fields.'''
import numpy as np
import healpy as hp


def get_theta(nside):
    """Define a HEALPix map with the value of θ in each pixel at the input `nside`
    
    Parameters
    ----------
    nside : int
        The `nside` of the map
        
    Returns
    -------
    theta : np.array
        A healpix map with the value of theta in each pixel, in radians.
        
    """
    theta, _ = hp.pix2ang(nside, np.arange(12 * nside ** 2))
    return np.pi/2. - theta

def healpix_derivatives(mapp, lmax=None, gradient=False, **kwargs):
    """Find the derivatives d_theta, d_phi of a Healpix map. It uses the healpy alm2map_der1 function.

    Parameters
    ----------
    mapp : np.array
        The Healpix map to find its derivatives.
    
    lmax : int, optional
        Maximum multipole to get the alm and the derivatives. It can create numerical errors if it is too high. Default: 3*nside-1.
        
    gradient : bool, optional
        If True, return the covariant derivatives. If False, return the partial derivatives. Default: False.
        
    **kwargs :
        Extra keywords to pass to the map2alm function.
        
    Returns
    -------
    d_theta : np.array
        A healpix map with the derivative with respect to theta.
    
    d_phi : np.array
        A healpix map with the derivatives with respect to phi, without normalizing.
        
    """
    nside = hp.get_nside(mapp)
    if lmax is None:
        lmax = 3 * nside - 1

    alm = hp.map2alm(mapp, lmax=lmax, **kwargs)
    [_, d_theta, d_phiosin] = hp.alm2map_der1(alm, nside, lmax=lmax)
    d_theta = -d_theta

    if gradient:
        return (d_theta, d_phiosin)
    
    theta = get_theta(nside)

    d_phi = d_phiosin * np.cos(theta)
    return (d_theta, d_phi)



def healpix_second_derivatives(d_theta, d_phi, lmax=None, **kwargs):
    """Find the second partial derivatives for every pixel of a Healpix map given the first partial derivatives.

    Parameters
    ----------
    d_theta : np.array
        The partial theta derivative of the Healpix map.
    
    d_phi : np.array
        The partial phi derivative of the Healpix map.
    
    lmax : int, optional
        Maximum multipole to get the alm and the derivatives. It can create numerical errors if it is too high. Default: 3*nside-1.
        
    Returns
    -------
    d_thetatheta : np.array
        A Healpix map of the second partial derivative wrt theta.
    
    d_phiphi : np.array
        A Healpix map of the second partial derivative wrt phi.
        
    d_phitheta : np.array
        A Healpix map of the second partial derivative wrt theta and phi.
        
    """
    nside = hp.get_nside(d_theta)

    theta = get_theta(nside)

    if lmax is None:
        lmax = 3 * nside - 1

    d_phitheta, d_phiphi = healpix_derivatives(d_phi, lmax=lmax, **kwargs)

    alm_theta = hp.map2alm(d_theta, lmax=lmax, **kwargs)
    [_, d_thetatheta, _] = hp.alm2map_der1(alm_theta, nside, lmax=lmax)
    d_thetatheta = -d_thetatheta

    
    return (d_thetatheta, d_phiphi, d_phitheta)

__all__ = ["get_theta", "healpix_derivatives", "healpix_second_derivatives"]


# This function is meant to be used as a callable class
# The class is initialized with two arrays, Q and U.
# The function __call__ returns the value of f at a given
# position psi.
class QUarray(np.ndarray):
    '''Array to store Q and U maps in the SO(3) formalism.

    Parameters
    ----------
    Q : np.array
        Values of the Q field. It can be an array of arrays to represent several maps.
        
    U : np.array
        Values of the U field. Must have the same shape as `Q`.
        
    Notes
    -----
    The class is a subclass of `np.ndarray`, so it can be used as a numpy array.
    CAREFUL: the multiplication of two SO3arrays is not the same as the multiplication of the $f$ that they represent. 
    In order to multiply two SO3arrays, you have to use the __call__ method, i.e. multiply the two SO3arrays after evaluating them at a given positions psi.
    however, you can multiply a SO3array with a scalar, or add or substract SO3arrays, and the result will be the correct SO3array
    '''
    def __new__(cls, Q, U):
        obj = np.asarray([Q,U]).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    def __call__(self, psi):
        '''Return the value of f at a given position psi.
        
        Parameters
        ----------
        psi : float or np.array
            The position at which to evaluate f. It must be broadcastable with the shape of `Q` and `U`.
            
        Returns
        -------
        f : np.array
            The value of f at the given position psi.
        '''
        return self[0]*np.cos(2.*psi) - self[1]*np.sin(2.*psi)
    
    def derpsi(self):
        '''Return the derivative of f with respect to psi, which can be exactly computed in the SO(3) formalism.

        Returns
        -------
        df : np.array
            The derivative of f with respect to psi.
        '''
        return QUarray(-2.*self[1], 2.*self[0])
    
    def modulus(self):
        '''Return the modulus of f.

        Returns
        -------
        fmod : np.array
            The modulus of f.
        '''
        return np.sqrt(self[0]**2 + self[1]**2)
    
    def pol_angle(self):
        '''Return the polarization angle of f.

        Returns
        -------
        pol_angle : np.array
            The polarization angle of f.
        '''
        angle = np.arctan2(-self[1], self[0])/2
        angle = angle + np.pi/2*(1-np.sign(self(angle)))/2  # I don't remember why we did this, but it seems to work
        angle[angle<0] = angle[angle<0] + np.pi
        return angle




__all__ = ['get_theta', 'healpix_derivatives', 'healpix_second_derivatives', 'QUarray']


# This function is meant to be used as a callable class
# The class is initialized with two arrays, Q and U.
# The function __call__ returns the value of f at a given
# position psi.
class QUarray(np.ndarray):
    '''Array to store Q and U maps in the SO(3) formalism.

    Parameters
    ----------
    Q : np.array
        Values of the Q field. It can be an array of arrays to represent several maps.
        
    U : np.array
        Values of the U field. Must have the same shape as `Q`.
        
    Notes
    -----
    The class is a subclass of `np.ndarray`, so it can be used as a numpy array.
    CAREFUL: the multiplication of two SO3arrays is not the same as the multiplication of the $f$ that they represent. 
    In order to multiply two SO3arrays, you have to use the __call__ method, i.e. multiply the two SO3arrays after evaluating them at a given positions psi.
    however, you can multiply a SO3array with a scalar, or add or substract SO3arrays, and the result will be the correct SO3array
    '''
    def __new__(cls, Q, U):
        obj = np.asarray([Q,U]).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    def __call__(self, psi):
        '''Return the value of f at a given position psi.
        
        Parameters
        ----------
        psi : float or np.array
            The position at which to evaluate f. It must be broadcastable with the shape of `Q` and `U`.
            
        Returns
        -------
        f : np.array
            The value of f at the given position psi.
        '''
        return self[0]*np.cos(2.*psi) - self[1]*np.sin(2.*psi)
    
    def derpsi(self):
        '''Return the derivative of f with respect to psi, which can be exactly computed in the SO(3) formalism.

        Returns
        -------
        df : np.array
            The derivative of f with respect to psi.
        '''
        return QUarray(-2.*self[1], 2.*self[0])
    
    def modulus(self):
        '''Return the modulus of f.

        Returns
        -------
        fmod : np.array
            The modulus of f.
        '''
        return np.sqrt(self[0]**2 + self[1]**2)
    
    def pol_angle(self):
        '''Return the polarization angle of f.

        Returns
        -------
        pol_angle : np.array
            The polarization angle of f.
        '''
        angle = np.arctan2(-self[1], self[0])/2
        angle = angle + np.pi/2*(1-np.sign(self(angle)))/2  # I don't remember why we did this, but it seems to work
        angle[angle<0] = angle[angle<0] + np.pi
        return angle




__all__ = ['get_theta', 'healpix_derivatives', 'healpix_second_derivatives', 'QUarray']

__docformat__ = "numpy"
