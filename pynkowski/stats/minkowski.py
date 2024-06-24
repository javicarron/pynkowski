'''This submodule contains the functions to compute the Minkowski Functionals of a field.'''
import numpy as np
from scipy.special import comb, gamma
from .utils_st import subsample_us, define_ubins
from ..data.base_da import DataField
from ..theory.base_th import TheoryField

try:
    from tqdm.auto import tqdm
except:
    tqdm = lambda x, disable: x
    print('tqdm not loaded')

def _MF_prefactor(d,j):
    """Compute the prefactor in the definition of Minkowski Functionals. This factor multiplies the integral of the curvatures.

    Parameters
    ----------
    d : int
        The dimension of the ambient space (e.g., 2 for the sphere).
        
    j : int
        The index of the Minkowski functional (`j>=1`).
    
    Returns
    -------
    prefactor : float
        The prefactor in the definition of MFs.
    
    """
    return 1./ (comb(d,j) * j * np.pi**(j/2.) / gamma(j/2. +1.))
    

def V0(field, us, edges=False, verbose=True):
    """Compute the first Minkowski Functional, $V_0$, normalized by the volume of the space.

    Parameters
    ----------
    field : DataField or TheoryField
        Field on which to compute $V_0$, which can be a theoretical field or a data field.
        
    us : np.array
        The thresholds where $V_0$ is computed.
        
    edges : bool, optional
        If False (default), the given `us` is assumed to be an array of uniformly distributed thresholds, which are taken as the central values of the bins.
        If True, input `us` is assumed to be a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform distributions of thresholds. 
        In the latter case, the effective thresholds are the central value of the given bins.
        
    verbose : bool, optional
        If True (default), progress bars are shown for the computations on data.
    
    Returns
    -------
    V0 : np.array
        The values of the first Minkowski Functional at the given thresholds.
    
    """
    us = np.atleast_1d(us)
    us, dus = define_ubins(us, edges)
    
    if isinstance(field, TheoryField):
        us = subsample_us(us, dus)
        try:
            return np.nanmean(field.V0(us), axis=1)
        except AttributeError:
            raise NotImplementedError(f"The theoretical expectation of V0 for {field.name} fields is not implemented. If you know an expression, please get in touch.")
        
    if isinstance(field, DataField):
        try:
            return field._V0(us, dus, verbose=verbose)
        except AttributeError:                
            stat = np.zeros_like(us)
            for ii in tqdm(np.arange(us.shape[0]), disable=not verbose):
                stat[ii] = np.mean((field.field > us[ii])[field.mask])
            return stat
        
    else:
        raise TypeError(f"The field must be either TheoryField or DataField (or a subclass).")
    

def V1(field, us, edges=False, verbose=True):
    """Compute the second Minkowski Functional, $V_1$, normalized by the volume of the space.

    Parameters
    ----------
    field : DataField or TheoryField
        Field on which to compute $V_1$, which can be a theoretical field or a data field.
        
    us : np.array
        The thresholds where $V_1$ is computed.
        
    edges : bool, optional
        If False (default), the given `us` is assumed to be an array of uniformly distributed thresholds, which are taken as the central values of the bins.
        If True, input `us` is assumed to be a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform distributions of thresholds. 
        In the latter case, the effective thresholds are the central value of the given bins.
        
    verbose : bool, optional
        If True (default), progress bars are shown for the computations on data.
    
    Returns
    -------
    V1 : np.array
        The values of the second Minkowski Functional at the given thresholds.
    
    """
    us = np.atleast_1d(us)
    us, dus = define_ubins(us, edges)
    
    if isinstance(field, TheoryField):
        us = subsample_us(us, dus)
        try:
            return np.nanmean(field.V1(us), axis=1)
        except AttributeError:
            raise NotImplementedError(f"The theoretical expectation of V1 for {field.name} fields is not implemented. If you know an expression, please get in touch.") from None
        
    elif isinstance(field, DataField):
        try:
            return field._V1(us, dus, verbose=verbose)
        except AttributeError:
            if field.first_der is None:
                field.get_first_der()
                
            stat = np.zeros_like(us)
            grad_modulus = np.sqrt(np.sum((field.first_der)**2., axis=0))
            for ii in tqdm(np.arange(us.shape[0]), disable=not verbose):
                this_mask = (us[ii] + dus[ii]/2. > field.field) & (us[ii] - dus[ii]/2. <= field.field)
                stat[ii] = np.mean((grad_modulus*this_mask/dus[ii])[field.mask])
            return _MF_prefactor(field.dim, 1) * stat
        
    else:
        raise TypeError(f"The field must be either TheoryField or DataField (or a subclass).")
        
        
  
def general_curvature(field, order):
    """Computes the symmetrised polynomial of the principal curvatures of order `order`, multiplied by the modulus of the gradient, at every pixel.

    Parameters
    ----------
    field : DataField
        Field on which to compute the general curvatures.
        
    order : int
        Order of the polynomial of the curvatures.
    
    Returns
    -------
    curvature : np.array
        The value of the polynomial at each pixel. 
        
        For example:
        `field.dim = 2`, `order=1` : geodesic curvature $\kappa$
        `field.dim = 3`, `order=1` : mean curvature (H = ($\kappa_1$ + $\kappa_2$)/2 )
        `field.dim = 3`, `order=2` : Gaussian curvature (K = $\kappa_1 \cdot \kappa_2$)
    
    """
    assert isinstance(field, DataField), "The field must be a DataField (or subclass)"
    assert field.dim > 1, "The field must be at least 2D"
    if order >= field.dim:
        raise ValueError("The order of the polynomial has to be less than the dimension of the field.")
        
    if order == 0:
        return np.sqrt(np.sum(field.first_der**2., axis=0))
    
    if field.dim == 2:
        if order == 1: # Geodesic curvature κ  (times |∇f|)
            num = 2.*field.first_der[0]*field.first_der[1]*field.second_der[2] - field.first_der[0]**2.*field.second_der[1] - field.first_der[1]**2.*field.second_der[0]
            den = field.first_der[0]**2. + field.first_der[1]**2.
            return num / den
            
    if field.dim == 3:
        if order == 1: # Mean curvature (times 2), 2H = κ_1 + κ_2   (times |∇f|)
            mod_grad = np.sqrt(np.sum(field.first_der**2., axis=0))
            hessian = np.array([[field.second_der[0], field.second_der[3], field.second_der[4]],
                                [field.second_der[3], field.second_der[1], field.second_der[5]],
                                [field.second_der[4], field.second_der[5], field.second_der[2]]])
            norm_grad = (field.first_der / mod_grad)
            mean_curvature = np.einsum('j...,jk...,k...->...', norm_grad, hessian, norm_grad) - np.trace(hessian, axis1=0, axis2=1)
            return mean_curvature
            
        if order == 2: # Gaussian curvature, K = κ_1 · κ_2   (times |∇f|)
            mod_grad = np.sqrt(np.sum(field.first_der**2., axis=0))
            norm_grad = (field.first_der / mod_grad)
            norm_second_der = (field.second_der / mod_grad)
            extended_hessian_det = np.linalg.det(np.array([[norm_second_der[0], norm_second_der[3], norm_second_der[4], norm_grad[0]],
                                                           [norm_second_der[3], norm_second_der[1], norm_second_der[5], norm_grad[1]],
                                                           [norm_second_der[4], norm_second_der[5], norm_second_der[2], norm_grad[2]],
                                                           [norm_grad[0], norm_grad[1], norm_grad[2], np.zeros_like(norm_grad[0])]]).T).T
            return -extended_hessian_det*mod_grad

        
    if field.dim > 3:
        raise NotImplementedError("The computation of principal curvatures in not implemented for spaces with 4 or more dimensions. If you need this, please get in touch.")
        # I am not sure if there is a easy expression for the principal curvatures of implicit surfaces in higher dimensions. Typically, one would need to create
        # the second fundamental form of the surface with the orthonormal bais of the tangent space; its eigenvalues are the principal curvatures at that point.
        

    
def V2(field, us, edges=False, verbose=True):
    """Compute the third Minkowski Functional, $V_2$, normalized by the volume of the space.

    Parameters
    ----------
    field : DataField or TheoryField
        Field on which to compute $V_2$, which can be a theoretical field or a data field. Its dimension must be at least 2.
        
    us : np.array
        The thresholds where $V_2$ is computed.
        
    edges : bool, optional
        If False (default), the given `us` is assumed to be an array of uniformly distributed thresholds, which are taken as the central values of the bins.
        If True, input `us` is assumed to be a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform distributions of thresholds. 
        In the latter case, the effective thresholds are the central value of the given bins.
        
    verbose : bool, optional
        If True (default), progress bars are shown for the computations on data.
    
    Returns
    -------
    V2 : np.array
        The values of the second Minkowski Functional at the given thresholds.
    
    """
    if field.dim<2:
        raise ValueError(f"V2 is defined for fields with at least 2 dimensions, but this field is {field.dim}D.")
        
    us = np.atleast_1d(us)
    us, dus = define_ubins(us, edges)
    
    if isinstance(field, TheoryField):
        us = subsample_us(us, dus)
        try:
            return np.nanmean(field.V2(us), axis=1)
        except AttributeError:
            raise NotImplementedError(f"The theoretical expectation of V2 for {field.name} fields is not implemented. If you know an expression, please get in touch.") from None
        
    elif isinstance(field, DataField):
        try:
            return field._V2(us, dus, verbose=verbose)
        except AttributeError:
            if field.first_der is None:
                field.get_first_der()
            if field.second_der is None:
                field.get_second_der()
                
            stat = np.zeros_like(us)
            curv = general_curvature(field, 1)
            for ii in tqdm(np.arange(us.shape[0]), disable=not verbose):
                this_mask = (us[ii] + dus[ii]/2. > field.field) & (us[ii] - dus[ii]/2. <= field.field)
                stat[ii] = np.nanmean((curv*this_mask/dus[ii])[field.mask])
            return _MF_prefactor(field.dim, 2) * stat
        
    else:
        raise TypeError(f"The field must be either TheoryField or DataField (or a subclass).")
        

def V3(field, us, edges=False, verbose=True):
    """Compute the third Minkowski Functional, $V_3$, normalized by the volume of the space.

    Parameters
    ----------
    field : DataField or TheoryField
        Field on which to compute $V_3$, which can be a theoretical field or a data field. Its dimension must be at least 3.
        
    us : np.array
        The thresholds where $V_3$ is computed.
        
    edges : bool, optional
        If False (default), the given `us` is assumed to be an array of uniformly distributed thresholds, which are taken as the central values of the bins.
        If True, input `us` is assumed to be a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform distributions of thresholds. 
        In the latter case, the effective thresholds are the central value of the given bins.
        
    verbose : bool, optional
        If True (default), progress bars are shown for the computations on data.
    
    Returns
    -------
    V3 : np.array()
        The values of the second Minkowski Functional at the given thresholds.
    
    """
    if field.dim<3:
        raise ValueError(f"V3 is defined for fields with at least 3 dimensions, but this field is {field.dim}D.")
        
    us = np.atleast_1d(us)
    us, dus = define_ubins(us, edges)
    
    if isinstance(field, TheoryField):
        us = subsample_us(us, dus)
        try:
            return np.nanmean(field.V3(us), axis=1)
        except AttributeError:
            raise NotImplementedError(f"The theoretical expectation of V3 for {field.name} fields is not implemented. If you know an expression, please get in touch.") from None
        
    elif isinstance(field, DataField):
        try:
            return field._V3(us, dus, verbose=verbose)
        except AttributeError:
            if field.first_der is None:
                field.get_first_der()
            if field.second_der is None:
                field.get_second_der()
                
            stat = np.zeros_like(us)
            curv = general_curvature(field, 2)
            for ii in tqdm(np.arange(us.shape[0]), disable=not verbose):
                this_mask = (us[ii] + dus[ii]/2. > field.field) & (us[ii] - dus[ii]/2. <= field.field)
                stat[ii] = np.nanmean((curv*this_mask/dus[ii])[field.mask])
            return _MF_prefactor(field.dim, 3) * stat
        
    else:
        raise TypeError(f"The field must be either TheoryField or DataField (or a subclass).")


__all__ = ["V0", "V1", "V2", "V3", "general_curvature"]

__docformat__ = "numpy"

