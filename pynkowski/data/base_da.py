'''This submodule contains the base abstract class for data fields.'''
import numpy as np


def _masked_variance(x, mask=None, weights=None):
    """Compute the variance of `x` within `mask`, weighting each pixel by `weights` if given.

    Parameters
    ----------
    x : np.array
        Values of the field.

    mask : np.array or None, optional
        Bool array of the same shape as `x`. If `None`, all pixels are included.

    weights : np.array or None, optional
        Weights of the pixels, of the same shape as `x`. If `None`, all pixels have the same weight.

    Returns
    -------
    var : float
        The (weighted) variance of `x` within `mask`.

    """
    if mask is not None:
        x = x[mask]
        weights = None if weights is None else weights[mask]
    if weights is None:
        return np.var(x)
    mean = np.average(x, weights=weights)
    return np.average((x - mean)**2., weights=weights)


def _prepare_pixel_weights(pixel_weights, shape):
    """Check the pixel weights and return them as a float array (or `None`)."""
    if pixel_weights is None:
        return None
    pixel_weights = np.asarray(pixel_weights, dtype=np.float64)
    if pixel_weights.shape != shape:
        raise ValueError(f"`pixel_weights` must have the same shape as the field, {shape}, not {pixel_weights.shape}.")
    if not (np.isfinite(pixel_weights).all() and (pixel_weights >= 0.).all()):
        raise ValueError("`pixel_weights` must be finite and non-negative.")
    return pixel_weights


class DataField():
    """General class for Data fields, to be used as base for all fields.

    Parameters
    ----------
    field : np.array
        Data of the field in an undefined structure.
    
    dim : int
        Dimension of the space where the field is defined.
        
    name : str, optional
        Name of the field.
        Defaul : `'DataField'`
    
    first_der : np.array or None, optional
        First **covariant** derivatives of the field in an orthonormal basis of the space. Same structure as `field`, and shape `(dim, field.shape)`.
    
    second_der : np.array or None, optional
        Second **covariant** derivatives of the field in an orthonormal basis of the space. Same structure as `field`, and shape `(dim*(dim+1)/2, field.shape)`.
        The order of the derivatives is diagonal first, e.g. in `dim=3`: `11`, `22`, `33`, `12`, `13`, `23`.
        
    mask : np.array or None, optional
        Mask where the field if considered. It is a bool array of the same shape that `field`.

    pixel_weights : np.array or None, optional
        Relative weight of each pixel in the averages over the mask, such as the area of the pixels for pixelisations with pixels of different sizes.
        It is an array of finite and non-negative values of the same shape that `field`.
        It is used by the computation of the Minkowski Functionals (except for fields with their own method, such as the $SO(3)$ fields) and by the normalisation of the field.
        It is not used by the maxima and minima distributions.
        Default: all pixels have the same weight.
        
    Attributes
    ----------
    field : np.array
        Data of the field in an undefined structure.
    
    dim : int
        Dimension of the space where the field is defined.
        
    name : str
        Name of the field.
    
    first_der : np.array or None
        First **covariant** derivatives of the field in an orthonormal basis of the space. Same structure as `field`, and shape `(dim, field.shape)`.
    
    second_der : np.array or None
        Second **covariant** derivatives of the field in an orthonormal basis of the space. Same structure as `field`, and shape `(dim*(dim+1)/2, field.shape)`.
        The order of the derivatives is diagonal first, e.g. in `dim=3`: `11`, `22`, `33`, `12`, `13`, `23`.
        
    mask : np.array
        Mask where the field if considered. It is a bool array of the same shape that `field`.

    pixel_weights : np.array or None
        Relative weight of each pixel in the averages over the mask, or `None` if all pixels have the same weight.
        
    """   
    
    def __init__(self, field, dim, name="DataField", first_der=None, second_der=None, mask=None, pixel_weights=None):
        self.field = field.copy()
        self.dim = dim
        self.name = name
        if mask is None:
            self.mask = np.ones_like(self.field, dtype='bool')
        else:
            self.mask = mask
        self.first_der = first_der
        self.second_der = second_der
        self.pixel_weights = _prepare_pixel_weights(pixel_weights, self.field.shape)
        
    def __repr__(self):
        return(f'{self.name} DataField: {self.field}')

    def _check_finite(self):
        """Raise a `ValueError` if the field is not finite."""
        if not np.isfinite(self.field).all():
            raise ValueError("The field must be finite (no NaN or inf). Please set the pixels outside of the region of interest to a finite value and exclude them with `mask`.")

    def _set_derivatives(self, first_der, second_der, scale=1.):
        """Check the given covariant derivatives, divide them by `scale`, and store them. If both are `None`, they are computed later when needed.

        Parameters
        ----------
        first_der : np.array or None
            First **covariant** derivatives of the field, with shape `(dim, field.shape)`.

        second_der : np.array or None
            Second **covariant** derivatives of the field, with shape `(dim*(dim+1)/2, field.shape)`.

        scale : float, optional
            The derivatives are divided by this value (e.g., the standard deviation used to normalise the field).
            Default: 1.

        """
        if (first_der is None) != (second_der is None):
            raise ValueError("`first_der` and `second_der` must be given together.")
        if first_der is not None:
            first_der, second_der = np.asarray(first_der), np.asarray(second_der)
            shape1 = (self.dim, *self.field.shape)
            shape2 = (self.dim * (self.dim + 1) // 2, *self.field.shape)
            if first_der.shape != shape1 or second_der.shape != shape2:
                raise ValueError(f"`first_der` and `second_der` must have shapes {shape1} and {shape2}, not {first_der.shape} and {second_der.shape}.")
            if not (np.isfinite(first_der).all() and np.isfinite(second_der).all()):
                raise ValueError("`first_der` and `second_der` must be finite (no NaN or inf).")
            if scale != 1.:
                first_der = first_der / scale
                second_der = second_der / scale
        self.first_der = first_der
        self.second_der = second_der
       


__all__ = ["DataField"]

__docformat__ = "numpy"
