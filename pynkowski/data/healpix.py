'''This submodule contains the class for scalar fields in HEALPix format and an easy interface for $P^2$ maps in this format.'''
import numpy as np
import healpy as hp
from .base_da import DataField
from .utils_da import get_theta, healpix_derivatives, healpix_second_derivatives

def _hotspot(healpix_map):
    """Find the local maxima of the input HEALPix map.

    Returns
    -------
    pixels : np.array
        Indices of the pixels which are local maxima.

    values : np.array
        Values of input map which are local maxima.

    """
    nside = hp.npix2nside(healpix_map.shape[0])
    neigh = hp.get_all_neighbours(nside, np.arange(12*nside**2))
    
    extT = np.concatenate([healpix_map, [np.min(healpix_map)-1.]])
    neigh_matrix = extT[neigh]

    mask = np.all(healpix_map > neigh_matrix, axis=0)
    pixels = np.argwhere(mask).flatten()
    values = healpix_map[pixels]
    return(pixels, values)


class Healpix(DataField):
    """Class for spherical scalar fields in HEALPix format.

    Parameters
    ----------
    field : np.array
        Values of the scalar field in HEALPix format in RING scheme.
        
    normalise : bool, optional
        If `True`, the map is normalised to have unit variance. Note: the mean of the map is not forced to be 0.
        Default: `True`.

    mask : np.array or None, optional
        Mask where the field is considered. It is a bool array of the same shape that `field`.
        Default: all data is included.
        
    Attributes
    ----------
    field : np.array
        Data of the field as a HEALPix map in RING scheme.
        
    nside : int
        Parameter `nside` of the map. The number of pixels is `12*nside**2`.
    
    dim : int
        Dimension of the space where the field is defined. In this case, the space is the sphere and this is 2.
        
    name : str
        Name of the field. In this case, "HEALPix map"
    
    first_der : np.array or None
        First **covariant** derivatives of the field in an orthonormal basis of the space. Same structure as `field`, and shape `(dim, field.shape)`.
    
    second_der : np.array or None
        Second **covariant** derivatives of the field in an orthonormal basis of the space. Same structure as `field`, and shape `(dim*(dim+1)/2, field.shape)`.
        The order of the derivatives is diagonal first, e.g. in `dim=3`: `11`, `22`, `33`, `12`, `13`, `23`.
        
    mask : np.array
        Mask where the field is considered. It is a bool array of the same shape that `field`.
        
    """   
    def __init__(self, field, normalise=True, mask=None):
        super().__init__(field, 2, name="HEALPix map", mask=mask)
        self.nside = hp.get_nside(self.field)
        if hp.get_nside(self.mask) != self.nside:
            raise ValueError('The map and the mask have different nside')
        
        if normalise:
            # self.field -= self.field.mean()
            σ2 = self.get_variance()
            self.field /= np.sqrt(σ2)
            
    def get_variance(self):
        """Compute the variance of the Healpix scalar map within the mask. 

        Returns
        -------
        var : float
            The variance of the Healpix map within the mask.

        """    
        return (np.var(self.field[self.mask]))
    
    def get_first_der(self, lmax=None):
        """Compute the covariant first derivatives of the input Healpix scalar map. 
        It stores:
        
        - first covariant derivative wrt θ in self.first_der[0]
        - first covariant derivative wrt ϕ in self.first_der[1]
        
        Parameters
        ----------
        lmax : int or None, optional
            Maximum multipole used in the computation of the derivatives.
            Default: 3*nside - 1
        """    
        self.first_der = np.array(healpix_derivatives(self.field, gradient=True, lmax=lmax))  # order: θ, ϕ
            
    def get_second_der(self, lmax=None):
        """Compute the covariant second derivatives of the input Healpix scalar map. 
        It stores:
        
        - second covariant derivative wrt θθ in self.second_der[0]
        - second covariant derivative wrt ϕϕ in self.second_der[1]
        - second covariant derivative wrt θϕ in self.second_der[2]

        Parameters
        ----------
        lmax : int or None, optional
            Maximum multipole used in the computation of the derivatives.
            Default: 3*nside - 1
        """    
        if self.first_der is None:
            self.get_first_der()
        theta = get_theta(self.nside)
        
        second_partial = healpix_second_derivatives(self.first_der[0], np.cos(theta) * self.first_der[1], lmax=lmax)  #order θθ, ϕϕ, θϕ
        
        self.second_der = np.array([second_partial[0],
                                    second_partial[1]/np.cos(theta)**2. + self.first_der[0]*np.tan(theta),
                                    (second_partial[2]/np.cos(theta) - self.first_der[1] * np.tan(theta))])  #order θθ, ϕϕ, θϕ

    def maxima_list(self):
        """Compute the values of the local maxima of the HEALPix map.

        Returns
        -------
        values : np.array
            Values of the map which are local maxima.

        """
        pixels, values = _hotspot(self.field)
        return values[self.mask[pixels]]

    def minima_list(self):
        """Compute the values of the local minima of the HEALPix map.

        Returns
        -------
        values : np.array
            Values of the map which are local minima.

        """
        pixels, values = _hotspot(-self.field)
        return -values[self.mask[pixels]]


class HealpixP2(Healpix):
    """Class for spherical scalar fields in HEALPix format.

    Parameters
    ----------
    Q : np.array
        Values of the Q component in HEALPix format in RING scheme.
        
    U : np.array
        Values of the U component in HEALPix format in RING scheme.
        
    normalise : bool, optional
        If `True`, Q and U are normalised to unit variance. Note: the normalisation is computed as the average of both variances, the mean of both maps are not forced to be 0.
    
    mask : np.array or None, optional
        Mask where the field if considered. It is a bool array of the same shape that `field`.
        Default: all data is included.
        
    Attributes
    ----------
    Q : np.array
        Data of the $Q$ component as a HEALPix map in RING scheme.
        
    U : np.array
        Data of the $U$ component as a HEALPix map in RING scheme.
        
    field : np.array
        Data of the $P^2$ field as a HEALPix map in RING scheme.
        
    nside : int
        Parameter `nside` of the map. The number of pixels is `12*nside**2`.
    
    dim : int
        Dimension of the space where the field is defined. In this case, the space is the sphere and this is 2.
        
    name : str
        Name of the field. In this case, "HEALPix map"
    
    first_der : np.array or None
        First **covariant** derivatives of the field in an orthonormal basis of the space. Same structure as `field`, and shape `(dim, field.shape)`.
    
    second_der : np.array or None
        Second **covariant** derivatives of the field in an orthonormal basis of the space. Same structure as `field`, and shape `(dim*(dim+1)/2, field.shape)`.
        The order of the derivatives is diagonal first, e.g. in `dim=3`: `11`, `22`, `33`, `12`, `13`, `23`.
        
    mask : np.array
        Mask where the field if considered. It is a bool array of the same shape that `field`.
        
    """   
    def __init__(self, Q, U, normalise=True, mask=None):
        self.Q = Q
        self.U = U
        self.mask = mask
        if normalise:
            # self.Q -= self.Q.mean()
            # self.U -= self.U.mean()
            σ2 = self.get_variance()
            self.Q /= np.sqrt(σ2)
            self.U /= np.sqrt(σ2)
        
        super().__init__(self.Q**2. + self.U**2., normalise=False, mask=mask)
            
    def get_variance(self):
        """Compute the variance of the input Healpix scalar map within the input mask. 

        Returns
        -------
        var : float
            The variance of the input Healpix map within the input mask.

        """    
        return (np.var(self.Q[self.mask]) + np.var(self.U[self.mask]))/2.
    

__all__ = ["Healpix", "HealpixP2"]
__docformat__ = "numpy"
