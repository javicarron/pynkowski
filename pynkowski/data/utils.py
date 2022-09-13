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

def derivatives(mapp, lmax=None, gradient=False, **kwargs):
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



def second_derivatives(d_theta, d_phi, lmax=None, **kwargs):
    """Find the Second derivatives for every pixel of a Healpix map given the first partial derivatives.

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

    d_phitheta, d_phiphi = derivatives(d_phi, lmax=lmax, **kwargs)

    alm_theta = hp.map2alm(d_theta, lmax=lmax, **kwargs)
    [_, d_thetatheta, _] = hp.alm2map_der1(alm_theta, nside, lmax=lmax)
    d_thetatheta = -d_thetatheta

    
    return (d_thetatheta, d_phiphi, d_phitheta)


__docformat__ = "numpy"
