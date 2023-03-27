import numpy as np
import healpy as hp
from .base_da import DataField
from .utils_da import get_theta, healpix_derivatives, healpix_second_derivatives


class Healpix(DataField):
    """Class for spherical scalar fields in HEALPix format.

    Parameters
    ----------
    field : np.array
        Values of the scalar field in HEALPix format in RING scheme.
        
    normalise : bool, optional
        If `True`, the map is normalise to have unit variance. Note: the mean of the map is not forced to be 0.
    
    mask : np.array or None, optional
        Mask where the field if considered. It is a bool array of the same shape that `field`.
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
        Mask where the field if considered. It is a bool array of the same shape that `field`.
        
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
        """Compute the variance of the input Healpix scalar map within the input mask. 

        Returns
        -------
        var : float
            The variance of the input Healpix map within the input mask.

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
        theta = get_theta(self.nside)
        
        # self.grad_theta = Scalar(S_grad[0], normalise=False)
        # self.der_phi = Scalar(np.cos(theta) * S_grad[1], normalise=False)
        # self.grad_phi = Scalar(S_grad[1], normalise=False)   
        
            
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
        
        # self.der_theta_theta = Scalar(S_der_der[0], normalise=False)
        # self.der_phi_phi = Scalar(S_der_der[1]/np.cos(theta)**2. + self.grad_theta.Smap*np.tan(theta), normalise=False)
        # self.der_theta_phi = Scalar((S_der_der[2]/np.cos(theta) - self.grad_phi.Smap * np.tan(theta)) , normalise=False)
        



__all__ = ["Healpix"]
__docformat__ = "numpy"
