"""Submodule with the base class for theoretical fields, `TheoryField`."""


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
        The derivative of the covariance function at the origin.
        Default : 1.
        
    lkc_ambient : np.array or None, optional
        The values for the Lipschitz–Killing Curvatures of the ambient space. If `None`, it is assumed that the volume is 1 and the rest is 0. This is exact for many spaces like Euclidean spaces or the sphere, and exact to leading order in μ for the rest.
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
        The derivative of the covariance function at the origin.
        
    lkc_ambient : np.array or None
        The values for the Lipschitz–Killing Curvatures of the ambient space.
        
    """   
    def __init__(self, dim, name='TheoryField', sigma=1., mu=1., lkc_ambient=None):
        self.name = name
        self.dim = dim
        self.sigma = sigma
        self.mu = mu
        self.lkc_ambient = lkc_ambient
        
    def __repr__(self):
        return(f'"{self.name}" TheoryField, {self.dim}D, σ = {self.sigma:.1f}, μ = {self.mu:.1f}')
        
      
