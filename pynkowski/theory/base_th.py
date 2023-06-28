"""This submodule contains the base abstract class for theoretical fields, `TheoryField`."""
import numpy as np

def _prepare_lkc(dim=None, lkc_ambient=None):
    """Define the Lipschitz–Killing Curvatures of the ambient manifold as the default ones (unit volume and the rest are 0), or verify their consistency. 
    If no argument is given, it defaults to a 2D space.

    Parameters
    ----------
    dim : int, optional
        The dimension of the ambient manifold.
        
    lkc_ambient : list, optional
        A list of the Lipschitz–Killing Curvatures of the ambient manifold. Its lenght must be `dim+1` if both arguments are given.
        
    Returns
    ----------
    dim : int 
        The dimension of the ambient manifold.
        
    lkc_ambient : np.array or None, optional
        An array of the Lipschitz–Killing Curvatures of the ambient manifold.
    """
    if lkc_ambient is None: 
        if dim is None:
            dim = 2
        lkc_ambient = np.zeros(dim+1)
        lkc_ambient[-1] = 1.
    else:
        if dim is None:
            dim = len(lkc_ambient) -1
        else:
            assert len(lkc_ambient) == dim +1, 'If both dim and lkc_ambient are given, len(lkc_ambient) == dim +1'
    return dim, lkc_ambient

class TheoryField():
    """General class for Theoretical fields, to be used as base for all fields.

    Parameters
    ----------
    dim : int
        Dimension of the space where the field is defined.
    
    name : str, optional
        Name of the field.
        Defaul : `'TheoryField'`
    
    sigma : float, optional
        The standard deviation of the field.
        Default : 1.
        
    mu : float, optional
        The derivative of the covariance function at the origin, times $-2$.
        Default : 1.
        
    nu : float, optional
        The second derivative of the covariance function at the origin.
        Default : 1.
        
    lkc_ambient : np.array or None, optional
        The values for the Lipschitz–Killing Curvatures of the ambient space. If `None`, it is assumed that the volume is 1 and the rest is 0. This (times the volume) is exact for many spaces like Euclidean spaces or the sphere, and exact to leading order in μ for the rest.
        Default : None
        
    Attributes
    ----------
    dim : int
        Dimension of the space where the field is defined.
    
    name : str
        Name of the field.
    
    sigma : float
        The standard deviation of the field.
        
    mu : float
        The derivative of the covariance function at the origin, times $-2$. Equal to the variance of the first derivatives of the field.

    nu : float
        The second derivative of the covariance function at the origin.
        
    lkc_ambient : np.array or None
        The values for the Lipschitz–Killing Curvatures of the ambient space.
        
    """   
    def __init__(self, dim, name='TheoryField', sigma=1., mu=1., nu=1., lkc_ambient=None):
        self.name = name
        self.sigma = sigma
        self.mu = mu
        self.nu = nu
        self.dim, self.lkc_ambient = _prepare_lkc(dim, lkc_ambient)
        
    def __repr__(self):
        return(f'"{self.name}" TheoryField, {self.dim}D, σ = {self.sigma:.1f}, μ = {self.mu:.1f}')
        
      
__all__ = ["TheoryField"]

__docformat__ = "numpy"

 