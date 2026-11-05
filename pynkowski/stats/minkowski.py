'''This submodule contains the functions to compute the Minkowski Functionals of a field.

For data fields, the Minkowski Functionals are computed with a single pass over the field: each pixel is assigned to its threshold bin,
and its contribution is accumulated with `np.bincount`. In this way, the computational cost does not grow with the number of thresholds.
The pixels can be processed in chunks of `chunk_voxels` pixels, in parallel with `threads` threads (by default, one per available CPU, limited by `OMP_NUM_THREADS` if it is set).

The data fields only need to provide the field, the mask, the covariant derivatives (or the methods `get_first_der` and `get_second_der`
to compute them), and optionally the weights of the pixels (`pixel_weights`). A data field can also define its own methods `_V0`, `_V1`,
`_V2` and `_V3` (e.g., the fields in the $SO(3)$ formalism, see `pynkowski.data.so3data`), which are then used instead of the general computation.
'''
import numbers
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.special import comb, gamma
from .utils_st import subsample_us, define_ubins
from ..data.base_da import DataField
from ..theory.base_th import TheoryField


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


def _curvature(order, g, h=None):
    """Compute the symmetrised polynomial of the principal curvatures of order `order`, multiplied by the modulus of the gradient,
    from the covariant derivatives of the field. The curvatures are computed in closed form, without building the Hessian matrix at every pixel.

    Parameters
    ----------
    order : int
        Order of the polynomial of the curvatures (`0` for the modulus of the gradient).

    g : list of np.array
        First covariant derivatives of the field, in the same order as `first_der`.

    h : list of np.array or None, optional
        Second covariant derivatives of the field, in the same order as `second_der`. Only used if `order>=1`.

    Returns
    -------
    curvature : np.array
        The value of the polynomial at each pixel. It is `nan` where the gradient vanishes (for `order>=1`).

    """
    dim = len(g)
    if order == 0:
        return np.sqrt(sum(gi * gi for gi in g))
    if dim == 2 and order == 1:  # Geodesic curvature κ  (times |∇f|)
        g1, g2 = g
        a, b, c = h
        return (2. * g1 * g2 * c - g1 * g1 * b - g2 * g2 * a) / (g1 * g1 + g2 * g2)
    if dim == 3 and order in (1, 2):
        gx, gy, gz = g
        a, b, c, d, e, f = h     # Order: 11, 22, 33, 12, 13, 23
        g2 = gx * gx + gy * gy + gz * gz
        if order == 1:  # Mean curvature (times 2), 2H = κ_1 + κ_2   (times |∇f|)
            return (gx * gx * a + gy * gy * b + gz * gz * c + 2. * (gx * gy * d + gx * gz * e + gy * gz * f)) / g2 - (a + b + c)
        # Gaussian curvature, K = κ_1 · κ_2   (times |∇f|). The gradient is contracted with the adjugate of the Hessian
        quad = (gx * gx * (b * c - f * f) + gy * gy * (a * c - e * e) + gz * gz * (a * b - d * d)
                + 2. * gx * gy * (e * f - d * c) + 2. * gx * gz * (d * f - b * e) + 2. * gy * gz * (d * e - a * f))
        return quad / np.sqrt(g2) ** 3
    if dim > 3:
        raise NotImplementedError("The computation of principal curvatures in not implemented for spaces with 4 or more dimensions. If you need this, please get in touch.")
        # I am not sure if there is a easy expression for the principal curvatures of implicit surfaces in higher dimensions. Typically, one would need to create
        # the second fundamental form of the surface with the orthonormal bais of the tangent space; its eigenvalues are the principal curvatures at that point.
    raise ValueError("The order of the polynomial has to be less than the dimension of the field.")


def general_curvature(field, order):
    """Computes the symmetrised polynomial of the principal curvatures of order `order`, multiplied by the modulus of the gradient, at every pixel.

    Parameters
    ----------
    field : DataField
        Field on which to compute the general curvatures. Its derivatives must have been computed.

    order : int
        Order of the polynomial of the curvatures.

    Returns
    -------
    curvature : np.array
        The value of the polynomial at each pixel.

        For example:
        `field.dim = 2`, `order=1` : geodesic curvature $\\kappa$
        `field.dim = 3`, `order=1` : mean curvature (H = ($\\kappa_1$ + $\\kappa_2$)/2 )
        `field.dim = 3`, `order=2` : Gaussian curvature (K = $\\kappa_1 \\cdot \\kappa_2$)
    
    """
    assert isinstance(field, DataField), "The field must be a DataField (or subclass)"
    assert field.dim > 1, "The field must be at least 2D"
    if order >= field.dim:
        raise ValueError("The order of the polynomial has to be less than the dimension of the field.")
    return _curvature(order, list(field.first_der), None if order == 0 else list(field.second_der))


def _env_thread_limit(name):
    """Return the number of threads set by the environment variable `name`, or `None` if it is not set or not valid.
    For `OMP_NUM_THREADS`, a comma-separated list (one value per nesting level) is accepted, and its first value is used."""
    value = os.environ.get(name, '').split(',')[0].strip()
    if value.isdigit() and int(value) > 0:
        return int(value)
    return None


def _available_cpus():
    """Return the default number of threads: the number of CPUs available to this process, limited by
    `OMP_NUM_THREADS` and `SLURM_CPUS_PER_TASK` if they are set."""
    try:
        n = len(os.sched_getaffinity(0))    # Linux
    except AttributeError:
        n = os.cpu_count() or 1             # macOS and Windows
    for name in ('OMP_NUM_THREADS', 'SLURM_CPUS_PER_TASK'):
        limit = _env_thread_limit(name)
        if limit is not None:
            n = min(n, limit)
    return max(n, 1)


def _map_chunks(fn, n, threads, chunk_voxels):
    """Apply `fn` to consecutive slices of `chunk_voxels` pixels (out of `n`), in parallel if `threads > 1`, and return the list of results.
    If `threads` is None, the number of threads is given by `_available_cpus`."""
    if threads is None:
        threads = _available_cpus()
    if not (isinstance(threads, numbers.Integral) and threads >= 1):
        raise ValueError(f"`threads` must be a positive integer or None, not {threads}.")
    if not (isinstance(chunk_voxels, numbers.Integral) and chunk_voxels >= 1):
        raise ValueError(f"`chunk_voxels` must be a positive integer, not {chunk_voxels}.")
    slices = [slice(i, min(i + chunk_voxels, n)) for i in range(0, n, chunk_voxels)]
    if threads > 1 and len(slices) > 1:
        with ThreadPoolExecutor(min(threads, len(slices))) as ex:
            return list(ex.map(fn, slices))
    return [fn(sl) for sl in slices]


def _flat_data(field):
    """Return the flattened field, mask and pixel weights (or `None`) of a data field, checking that the field is finite."""
    f = np.reshape(field.field, -1)
    if not np.isfinite(f).all():
        raise ValueError("The field must be finite (no NaN or inf). Please set the pixels outside of the region of interest to a finite value and exclude them with `mask`.")
    m = np.reshape(np.asarray(field.mask, dtype=bool), -1)
    if m.shape != f.shape:
        raise ValueError('The field and the mask have different shapes')
    w = getattr(field, 'pixel_weights', None)
    if w is not None:
        w = np.reshape(w, -1)
        if w.shape != f.shape:
            raise ValueError('The field and `pixel_weights` have different shapes')
    return f, m, w


def _V0_data(field, us, threads, chunk_voxels):
    """Compute $V_0$ of a data field at the thresholds `us`, which can be in any order, with a single pass over the field."""
    f, m, w = _flat_data(field)
    us = np.asarray(us, dtype=np.float64)
    # The thresholds are sorted for `np.searchsorted`, and the result is given back in the original order
    order = np.argsort(us, kind='stable')
    us_sorted = us[order]

    def process(sl):
        sel = m[sl]
        fl = np.asarray(f[sl][sel], dtype=np.float64)
        ww = None if w is None else np.asarray(w[sl][sel], dtype=np.float64)
        # For each pixel, we count the number of thresholds strictly below its value
        counts = np.bincount(np.searchsorted(us_sorted, fl, side='left'), weights=ww, minlength=len(us) + 1)
        return counts, (fl.size if w is None else ww.sum())

    results = _map_chunks(process, f.size, threads, chunk_voxels)
    counts = sum(r[0] for r in results)
    total = sum(r[1] for r in results)
    v0 = np.empty(len(us))
    # The pixels above `us_sorted[i]` are those with more than `i` thresholds below their value
    v0[order] = np.cumsum(counts[::-1])[::-1][1:] / total
    return v0


def _Vj_data(field, j, us, dus, threads, chunk_voxels):
    """Compute the Minkowski Functional $V_j$ (`j>=1`) of a data field in the bins of centres `us` and widths `dus`, with a single pass over the field."""
    order = j - 1    # Order of the polynomial of the curvatures
    if order >= 1 and field.dim > 3:
        raise NotImplementedError("The computation of principal curvatures in not implemented for spaces with 4 or more dimensions. If you need this, please get in touch.")
    if field.first_der is None:
        field.get_first_der()
    if order >= 1 and field.second_der is None:
        field.get_second_der()

    f, m, w = _flat_data(field)
    us = np.asarray(us, dtype=np.float64)
    dus = np.asarray(dus, dtype=np.float64)
    edges = np.append(us - dus / 2., us[-1] + dus[-1] / 2.)
    if not np.all(np.diff(edges) > 0):
        raise ValueError("The thresholds `us` must be increasing.")
    nb = len(us)
    fd = np.reshape(field.first_der, (field.dim, -1))
    sd = np.reshape(field.second_der, (-1, f.size)) if order >= 1 else None

    def process(sl):
        idx = np.digitize(np.asarray(f[sl], dtype=np.float64), edges) - 1
        mm = m[sl]
        c = _curvature(order, [np.asarray(g[sl], dtype=np.float64) for g in fd],
                       None if sd is None else [np.asarray(h[sl], dtype=np.float64) for h in sd])
        isnan = np.isnan(c)
        if w is not None:
            ww = np.asarray(w[sl], dtype=np.float64)
            c = c * ww
        # The pixels outside of the bins or of the mask, or with an undefined curvature, go to an extra bin that is discarded
        bad = (idx < 0) | (idx >= nb) | ~mm | isnan
        tot = np.bincount(np.where(bad, nb, idx), weights=np.where(isnan, 0., c), minlength=nb + 1)[:nb]
        if w is None:
            return tot, np.count_nonzero(mm & isnan), np.count_nonzero(mm)
        return tot, ww[mm & isnan].sum(), ww[mm].sum()

    results = _map_chunks(process, f.size, threads, chunk_voxels)
    tot = sum(r[0] for r in results)
    nan_total = sum(r[1] for r in results)
    mask_total = sum(r[2] for r in results)
    # The pixels with an undefined curvature (|∇f| = 0) are excluded from the average for V2 and V3 (as with `np.nanmean`), but not for V1
    denom = mask_total if order == 0 else mask_total - nan_total
    return _MF_prefactor(field.dim, j) * tot / (dus * denom)


def V0(field, us, edges=False, verbose=True, threads=None, chunk_voxels=1 << 17):
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
        If True (default), progress bars are shown for the computations on data fields that support them (currently, the $SO(3)$ fields).

    threads : int or None, optional
        Number of threads used for the computation on data fields. Only used by the general computation (i.e., not by fields with their own method, such as the $SO(3)$ fields).
        Default: None, i.e., one thread per CPU available to the process, limited by the environment variables `OMP_NUM_THREADS` and `SLURM_CPUS_PER_TASK` if they are set.

    chunk_voxels : int, optional
        Number of pixels processed at once by each thread in the computation on data fields. Lower values reduce the memory usage.
        The peak memory grows as `threads*chunk_voxels`. Only used by the general computation.
        Default: 2**17.

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
        
    elif isinstance(field, DataField):
        if hasattr(field, '_V0'):
            return field._V0(us, dus, verbose=verbose)
        return _V0_data(field, us, threads, chunk_voxels)
        
    else:
        raise TypeError(f"The field must be either TheoryField or DataField (or a subclass).")
    

def V1(field, us, edges=False, verbose=True, threads=None, chunk_voxels=1 << 17):
    """Compute the second Minkowski Functional, $V_1$, normalized by the volume of the space.

    Parameters
    ----------
    field : DataField or TheoryField
        Field on which to compute $V_1$, which can be a theoretical field or a data field.
        
    us : np.array
        The thresholds where $V_1$ is computed. For data fields, they must be increasing.
        
    edges : bool, optional
        If False (default), the given `us` is assumed to be an array of uniformly distributed thresholds, which are taken as the central values of the bins.
        If True, input `us` is assumed to be a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform distributions of thresholds. 
        In the latter case, the effective thresholds are the central value of the given bins.
        
    verbose : bool, optional
        If True (default), progress bars are shown for the computations on data fields that support them (currently, the $SO(3)$ fields).

    threads : int or None, optional
        Number of threads used for the computation on data fields. Only used by the general computation (i.e., not by fields with their own method, such as the $SO(3)$ fields).
        Default: None, i.e., one thread per CPU available to the process, limited by the environment variables `OMP_NUM_THREADS` and `SLURM_CPUS_PER_TASK` if they are set.

    chunk_voxels : int, optional
        Number of pixels processed at once by each thread in the computation on data fields. Lower values reduce the memory usage.
        The peak memory grows as `threads*chunk_voxels`. Only used by the general computation.
        Default: 2**17.

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
        if hasattr(field, '_V1'):
            return field._V1(us, dus, verbose=verbose)
        return _Vj_data(field, 1, us, dus, threads, chunk_voxels)
        
    else:
        raise TypeError(f"The field must be either TheoryField or DataField (or a subclass).")
        

def V2(field, us, edges=False, verbose=True, threads=None, chunk_voxels=1 << 17):
    """Compute the third Minkowski Functional, $V_2$, normalized by the volume of the space.

    Parameters
    ----------
    field : DataField or TheoryField
        Field on which to compute $V_2$, which can be a theoretical field or a data field. Its dimension must be at least 2.
        
    us : np.array
        The thresholds where $V_2$ is computed. For data fields, they must be increasing.
        
    edges : bool, optional
        If False (default), the given `us` is assumed to be an array of uniformly distributed thresholds, which are taken as the central values of the bins.
        If True, input `us` is assumed to be a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform distributions of thresholds. 
        In the latter case, the effective thresholds are the central value of the given bins.
        
    verbose : bool, optional
        If True (default), progress bars are shown for the computations on data fields that support them (currently, the $SO(3)$ fields).

    threads : int or None, optional
        Number of threads used for the computation on data fields. Only used by the general computation (i.e., not by fields with their own method, such as the $SO(3)$ fields).
        Default: None, i.e., one thread per CPU available to the process, limited by the environment variables `OMP_NUM_THREADS` and `SLURM_CPUS_PER_TASK` if they are set.

    chunk_voxels : int, optional
        Number of pixels processed at once by each thread in the computation on data fields. Lower values reduce the memory usage.
        The peak memory grows as `threads*chunk_voxels`. Only used by the general computation.
        Default: 2**17.

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
        if hasattr(field, '_V2'):
            return field._V2(us, dus, verbose=verbose)
        return _Vj_data(field, 2, us, dus, threads, chunk_voxels)
        
    else:
        raise TypeError(f"The field must be either TheoryField or DataField (or a subclass).")
        

def V3(field, us, edges=False, verbose=True, threads=None, chunk_voxels=1 << 17):
    """Compute the fourth Minkowski Functional, $V_3$, normalized by the volume of the space.

    Parameters
    ----------
    field : DataField or TheoryField
        Field on which to compute $V_3$, which can be a theoretical field or a data field. Its dimension must be at least 3.
        
    us : np.array
        The thresholds where $V_3$ is computed. For data fields, they must be increasing.
        
    edges : bool, optional
        If False (default), the given `us` is assumed to be an array of uniformly distributed thresholds, which are taken as the central values of the bins.
        If True, input `us` is assumed to be a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform distributions of thresholds. 
        In the latter case, the effective thresholds are the central value of the given bins.
        
    verbose : bool, optional
        If True (default), progress bars are shown for the computations on data fields that support them (currently, the $SO(3)$ fields).

    threads : int or None, optional
        Number of threads used for the computation on data fields. Only used by the general computation (i.e., not by fields with their own method, such as the $SO(3)$ fields).
        Default: None, i.e., one thread per CPU available to the process, limited by the environment variables `OMP_NUM_THREADS` and `SLURM_CPUS_PER_TASK` if they are set.

    chunk_voxels : int, optional
        Number of pixels processed at once by each thread in the computation on data fields. Lower values reduce the memory usage.
        The peak memory grows as `threads*chunk_voxels`. Only used by the general computation.
        Default: 2**17.

    Returns
    -------
    V3 : np.array()
        The values of the fourth Minkowski Functional at the given thresholds.
    
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
        if hasattr(field, '_V3'):
            return field._V3(us, dus, verbose=verbose)
        return _Vj_data(field, 3, us, dus, threads, chunk_voxels)
        
    else:
        raise TypeError(f"The field must be either TheoryField or DataField (or a subclass).")


__all__ = ["V0", "V1", "V2", "V3", "general_curvature"]

__docformat__ = "numpy"
