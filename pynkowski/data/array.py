import numpy as np
from .base_da import DataField



def _eval_field(field, pix):
    return field[tuple(pix[:,ii] for ii in np.arange(field.ndim))]

def _hotspot_array(field):
    """Find the local maxima of the input field.

    Arguments
    ---------
    field : np.array
        Array of the pixelized field values.

    Returns
    -------
    pixels : np.array
        Indices of the pixels which are local maxima.

    values : np.array
        Values of input map which are local maxima.

    """
    ndim = len(field.shape)
    # First we shift the field in each direction by 1 pixel, and check that the original pixel is larger than all the `2**ndim` neighbours.
    max_mask = np.all(field > np.array([np.roll(field, shift, axis=ii) for shift in [-1,1] for ii in range(ndim)]), axis=0)

    # We then remove all the pixels in the border to remove the edge effects.
    for dim in range(max_mask.ndim):
        # Make current dimension the first dimension
        array_moved = np.moveaxis(max_mask, dim, 0)
        # Set border values to `False` in the current dimension
        array_moved[0] = False
        array_moved[-1] = False
        # No need to reorder the dimensions as moveaxis returns a view
    max_mask = np.pad(max_mask, pad_width=1, mode='constant', constant_values=False)

    pixels = np.argwhere(max_mask)
    values = _eval_field(field, pixels)
    return(pixels, values)

class DataArray(DataField):
    """Class for a pixelized Euclidean data array.
    
    Parameters
    ----------
    field : np.array
        Pixelized field as an array of arbitrary dimension. All pixels (or voxels) are expected to be the same size. The pixel shape is not necessarily a cube (see `spacing`).
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
        Data of the field as an array.
        
    dim : int
        Dimension of the space where the field is defined. For example, 2 for an image and 3 for a data cube..
        
    name : str
        Name of the field. In this case, "DataArray"
        
    first_der : np.array or None
        First **covariant** derivatives of the field in an orthonormal basis of the space. Same structure as `field`, and shape `(dim, field.shape)`.
        
    second_der : np.array or None
        Second **covariant** derivatives of the field in an orthonormal basis of the space. Same structure as `field`, and shape `(dim*(dim+1)/2, field.shape)`.

    spacing : float or np.array
        Spacing between pixels (centres) in each dimension. If a float, the spacing is the same in all dimensions.
        If an array, it must have the same length as the number of dimensions of the field.
        
    """
    def __init__(self, field, normalise=True, mask=None, spacing=1.):
        dim = len(field.shape)
        super().__init__(field, dim=dim, name='DataArray', mask=mask)
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
        It stores the derivatives in order. E.g., in a 3D array:
        
        - first covariant derivative wrt e₁ in self.first_der[0]
        - first covariant derivative wrt e₂ in self.first_der[1]
        - first covariant derivative wrt e₃ in self.first_der[2]

        """
        self.first_der = np.array(np.gradient(self.field, self.spacing, edge_order=2))

    def get_second_der(self):
        """Compute the covariant second derivatives of the field. 
        It stores the second derivatives in the following order: first, all the diagonal derivatives, then the cross derivatives with e₁, then with e₂, and so on.
        E.g., in a 3D array:
        
        - second covariant derivative wrt e₁e₁ in self.second_der[0]
        - second covariant derivative wrt e₂e₂ in self.second_der[1]
        - second covariant derivative wrt e₃e₃ in self.second_der[2]
        - second covariant derivative wrt e₁e₂ in self.second_der[3]
        - second covariant derivative wrt e₁e₃ in self.second_der[4]
        - second covariant derivative wrt e₂e₃ in self.second_der[5]

        """
        if self.first_der is None:
            self.get_first_der()
        self.second_der = np.zeros((self.dim*(self.dim+1)//2, *self.field.shape))
        d = self.dim
        for i in np.arange(self.dim, dtype=int):
            indeces = [i]
            for j in np.arange(i+1, self.dim, dtype=int):
                indeces.append((d*(d-1)/2) - (d-i)*((d-i)-1)/2 + j - i - 1 + d)
            indeces = np.array(indeces, dtype=int)
            self.second_der[indeces] = np.gradient(self.first_der[i], self.spacing, edge_order=2)[i:d]

    def maxima_list(self):
        """Find the local maxima of the field.

        Returns
        -------
        values : np.array
            Values of input map which are local maxima.

        """
        pixels, values = _hotspot_array(self.field)
        return values[_eval_field(self.mask, pixels)]
    
    def minima_list(self):
        """Find the local minima of the field.

        Returns
        -------
        values : np.array
            Values of input map which are local minima.

        """
        pixels, values = _hotspot_array(-self.field)
        return -values[_eval_field(self.mask, pixels)]



__all__ = ["DataArray"]
__docformat__ = "numpy"
