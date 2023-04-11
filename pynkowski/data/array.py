import numpy as np
from .base_da import DataField

def _hotspot3D(field):
    """Find the local maxima of the input field.

    Arguments
    ---------
    field : np.array
        3D array of the pixelized field values.

    Returns
    -------
    pixels : np.array
        Indices of the pixels which are local maxima.

    values : np.array
        Values of input map which are local maxima.

    """
    # First we shift the field in each direction by 1 pixel, and check that the original pixel is larger than all the 8 neighbours.
    max_mask = np.all(field > np.array([np.roll(field, shift, axis=ii) for shift in [-1,1] for ii in range(3)]), axis=0)

    # We then remove all the pixels in the border to remove the edge effects.
    max_mask[0] = False
    max_mask[-1] = False
    max_mask[:,0] = False
    max_mask[:,-1] = False
    max_mask[:,:,0] = False
    max_mask[:,:,-1] = False

    pixels = np.argwhere(max_mask)
    values = field[pixels]
    return(pixels, values)

class DataCube(DataField):
    """Class for a pixelized Euclidean data cube.
    
    Parameters
    ----------
    field : np.array
        Pixelized field as a 3D array. All pixels (or voxels) are expected to be the same size. The pixel shape is not necessarily a cube (see `spacing`).
        The field can have different sizes in each dimension.
        
    normalise : bool, optional
        If `True`, the map is normalised to have unit variance. Note: the mean of the map is not forced to be 0.
        Default: `True`.
        
    mask : np.array or None, optional
        Mask where the field if considered. It is a bool array of the same shape that `field`.
        Default: all data is included.

    spacing : float or np.array, optional
        Spacing between pixels (centres) in each dimension. If a float, the spacing is the same in all dimensions. 
        If an array, it must have the same length as the number of dimensions of the field.
        
    Attributes
    ----------
    field : np.array
        Data of the field as a 3D array.
        
    dim : int
        Dimension of the space where the field is defined. In this case, the space is the 3D space and this is 3.
        
    name : str
        Name of the field. In this case, "data cube"
        
    first_der : np.array or None
        First **covariant** derivatives of the field in an orthonormal basis of the space. Same structure as `field`, and shape `(dim, field.shape)`.
        
    second_der : np.array or None
        Second **covariant** derivatives of the field in an orthonormal basis of the space. Same structure as `field`, and shape `(dim*(dim+1)/2, field.shape)`.
        
    """
    def __init__(self, field, normalise=True, mask=None, spacing=1.):
        super().__init__(field, dim=3, name='DataCube', mask=mask)
        if field.ndim != 3:
            raise ValueError('The field must be a 3D array')
        if self.mask.shape != self.field.shape:
            raise ValueError('The map and the mask have different shapes')
        if normalise:
            self.field /= np.sqrt(self.get_variance())
        self.first_der = None
        self.second_der = None
        self.spacing = spacing
            
    def get_variance(self):
        """Compute the variance of the field within the mask. 

        Returns
        -------
        var : float
            The variance of the input field within the mask.

        """    
        return (np.var(self.field[self.mask]))

    def get_first_der(self):
        """Compute the covariant first derivatives of the field. 
        It stores:
        
        - first covariant derivative wrt e₁ in self.first_der[0]
        - first covariant derivative wrt e₂ in self.first_der[1]
        - first covariant derivative wrt e₃ in self.first_der[2]

        """
        self.first_der = np.array(np.gradient(self.field, self.spacing, edge_order=2))

    def get_second_der(self):
        """Compute the covariant second derivatives of the field. 
        It stores:
        
        - second covariant derivative wrt e₁e₁ in self.second_der[0]
        - second covariant derivative wrt e₂e₂ in self.second_der[1]
        - second covariant derivative wrt e₃e₃ in self.second_der[2]
        - second covariant derivative wrt e₁e₂ in self.second_der[3]
        - second covariant derivative wrt e₁e₃ in self.second_der[4]
        - second covariant derivative wrt e₂e₃ in self.second_der[5]

        
        """
        self.second_der = np.zeros((self.dim*(self.dim+1)//2, *self.field.shape))
        self.second_der[[0,3,4]] = np.gradient(self.first_der[0], self.spacing, edge_order=2)
        self.second_der[[1,5]] = np.gradient(self.first_der[1], self.spacing, edge_order=2)[1:]
        self.second_der[2] = np.gradient(self.first_der[2], self.spacing, edge_order=2)[2]

    def maxima_list(self):
        """Find the local maxima of the field.

        Returns
        -------
        values : np.array
            Values of input map which are local maxima.

        """
        values, pixels = _hotspot3D(self.field)
        return values[self.mask[pixels]]
    
    def minima_list(self):
        """Find the local minima of the field.

        Returns
        -------
        values : np.array
            Values of input map which are local minima.

        """
        values, pixels = _hotspot3D(-self.field)
        return -values[self.mask[pixels]]



__all__ = ["DataCube"]
__docformat__ = "numpy"
