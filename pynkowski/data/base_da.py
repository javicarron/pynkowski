'''This submodule contains the base abstract class for data fields.'''
import numpy as np

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
        
    """   
    
    def __init__(self, field, dim, name="DataField", first_der=None, second_der=None, mask=None):
        self.field = field.copy()
        self.dim = dim
        self.name = name
        if mask is None:
            self.mask = np.ones_like(self.field, dtype='bool')
        else:
            self.mask = mask
        self.first_der = first_der
        self.second_der = second_der
        
    def __repr__(self):
        return(f'{self.name} DataField: {self.field}')
       


__all__ = ["DataField"]

__docformat__ = "numpy"
