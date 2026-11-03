'''This submodule contains the computation of the Minkowski Functionals of scalar data fields, used by `DataArray`, `Healpix` and `HealpixP2`.

The Minkowski Functionals are computed with a single pass over the field: each pixel is assigned to its threshold bin, and its contribution is accumulated with `np.bincount`. The result is the same as with the loop over the thresholds in `pynkowski.stats.minkowski` (up to floating-point rounding), but the computational cost does not grow with the number of thresholds.
'''
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ..stats.minkowski import _MF_prefactor


def _weights(order, g, h):
    """Compute the integrand of the Minkowski Functional $V_j$, with `j=order`, at every pixel. This is the modulus of the gradient for `order=1`, and the symmetrised polynomial of the principal curvatures of order `order-1`, multiplied by the modulus of the gradient, for `order>=2` (see `pynkowski.stats.minkowski.general_curvature`).
    The curvatures are computed in closed form, without building the Hessian matrix at every pixel.

    Parameters
    ----------
    order : int
        The index of the Minkowski Functional (`order>=1`).

    g : list
        First covariant derivatives of the field, in the same order as `first_der`.

    h : list or None
        Second covariant derivatives of the field, in the same order as `second_der`. Only used if `order>=2`.

    Returns
    -------
    weights : np.array
        The value of the integrand at each pixel.

    """
    if order == 1:
        return np.sqrt(sum(gi * gi for gi in g))
    if len(g) == 2:  # Geodesic curvature κ  (times |∇f|)
        g1, g2 = g
        a, b, c = h
        return (2 * g1 * g2 * c - g1 * g1 * b - g2 * g2 * a) / (g1 * g1 + g2 * g2)
    gx, gy, gz = g
    a, b, c, d, e, f = h
    g2 = gx * gx + gy * gy + gz * gz
    if order == 2:  # Mean curvature (times 2), 2H = κ_1 + κ_2   (times |∇f|)
        return (gx * gx * a + gy * gy * b + gz * gz * c + 2 * (gx * gy * d + gx * gz * e + gy * gz * f)) / g2 - (a + b + c)
    # Gaussian curvature, K = κ_1 · κ_2   (times |∇f|). The gradient is contracted with the adjugate of the Hessian
    quad = (gx * gx * (b * c - f * f) + gy * gy * (a * c - e * e) + gz * gz * (a * b - d * d)
            + 2 * gx * gy * (e * f - d * c) + 2 * gx * gz * (d * f - b * e) + 2 * gy * gz * (d * e - a * f))
    return quad / np.sqrt(g2) ** 3


class _FastMFMixin:
    """Class with the computation of the Minkowski Functionals of scalar data fields, to be used as a base for `DataArray` and `Healpix`.
    It implements `_V0`, `_V1`, `_V2` and `_V3`, the internal interfaces for `pynkowski.V0`, `pynkowski.V1`, `pynkowski.V2` and `pynkowski.V3`.

    The class using it must have it before `DataField` in its bases, and call `_init_fast` in its `__init__` once `field`, `mask` and `dim` are defined.

    """

    def _init_fast(self, normalise, first_der, second_der, threads, chunk_voxels):
        """Check the field and the given derivatives, normalise them if needed, and store the parameters for the computation of the Minkowski Functionals.

        Parameters
        ----------
        normalise : bool
            If `True`, the map (and `first_der` and `second_der`, if given) is normalised to have unit variance.

        first_der : np.array or None
            First **covariant** derivatives of the field, or `None` to compute them when needed.

        second_der : np.array or None
            Second **covariant** derivatives of the field, or `None` to compute them when needed.

        threads : int
            Number of threads used in the computation of the Minkowski Functionals.

        chunk_voxels : int
            Number of pixels processed at once by each thread.

        """
        if not np.isfinite(self.field).all():
            raise ValueError("The field must be finite (no NaN or inf). Please set the pixels outside of the region of interest to a finite value and exclude them with `mask`.")
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
        if normalise:
            sigma = np.sqrt(self.get_variance())
            self.field /= sigma
            if first_der is not None:
                first_der = first_der / sigma
                second_der = second_der / sigma
        self.first_der = first_der
        self.second_der = second_der
        self.threads = threads
        self.chunk_voxels = chunk_voxels

    def _map_chunks(self, fn, n):
        """Apply `fn` to consecutive slices of `chunk_voxels` pixels (out of `n`), in parallel if `threads > 1`, and return the list of results."""
        slices = [slice(i, min(i + self.chunk_voxels, n)) for i in range(0, n, self.chunk_voxels)]
        if self.threads and self.threads > 1 and len(slices) > 1:
            with ThreadPoolExecutor(self.threads) as ex:
                return list(ex.map(fn, slices))
        return [fn(sl) for sl in slices]

    def _v0(self, us):
        """Compute $V_0$ at the thresholds `us`, which can be in any order."""
        us = np.asarray(us, dtype=np.float64)
        # The thresholds are sorted for `np.searchsorted`, and the result is given back in the original order
        order = np.argsort(us, kind='stable')
        us_sorted = us[order]
        f = self.field.reshape(-1)
        m = self.mask.reshape(-1)

        def process(sl):
            fl = f[sl][m[sl]].astype(np.float64)
            # For each pixel, we count the number of thresholds strictly below its value
            return np.bincount(np.searchsorted(us_sorted, fl, side='left'), minlength=len(us) + 1), fl.size

        results = self._map_chunks(process, f.size)
        counts = sum(r[0] for r in results)
        v0 = np.empty(len(us))
        # The pixels above `us_sorted[i]` are those with more than `i` thresholds below their value
        v0[order] = np.cumsum(counts[::-1])[::-1][1:] / sum(r[1] for r in results)
        return v0

    def _mf(self, order, us, dus):
        """Compute the Minkowski Functional $V_j$, with `j=order>=1`, in the bins of centres `us` and widths `dus`."""
        if order > self.dim:
            raise ValueError(f"V{order} is defined for fields with at least {order} dimensions, but this field is {self.dim}D.")
        if order >= 2 and self.dim > 3:
            raise NotImplementedError("The computation of principal curvatures is not implemented for spaces with 4 or more dimensions. If you need this, please get in touch.")
        if self.first_der is None:
            self.get_first_der()
        if order >= 2 and self.second_der is None:
            self.get_second_der()

        us = np.asarray(us, dtype=np.float64)
        dus = np.asarray(dus, dtype=np.float64)
        edges = np.append(us - dus / 2., us[-1] + dus[-1] / 2.)
        if not np.all(np.diff(edges) > 0):
            raise ValueError("The thresholds `us` must be increasing.")
        nb = len(us)
        f = self.field.reshape(-1)
        m = self.mask.reshape(-1)
        fd = np.reshape(self.first_der, (self.dim, -1))
        sd = np.reshape(self.second_der, (-1, f.size)) if order >= 2 else None

        def process(sl):
            idx = np.digitize(f[sl].astype(np.float64), edges) - 1
            mm = m[sl]
            w = _weights(order, [g[sl].astype(np.float64) for g in fd],
                         None if sd is None else [h[sl].astype(np.float64) for h in sd])
            isnan = np.isnan(w)
            # The pixels outside of the bins or of the mask, or with an undefined curvature, go to an extra bin that is discarded
            bad = (idx < 0) | (idx >= nb) | ~mm | isnan
            tot = np.bincount(np.where(bad, nb, idx), weights=np.where(isnan, 0., w), minlength=nb + 1)[:nb]
            return tot, np.count_nonzero(mm & isnan), np.count_nonzero(mm)

        results = self._map_chunks(process, f.size)
        tot = sum(r[0] for r in results)
        nan_count = sum(r[1] for r in results)
        mask_count = sum(r[2] for r in results)
        # As with `np.mean` in `pynkowski.V1` and `np.nanmean` in `pynkowski.V2` and `pynkowski.V3`, the pixels with an undefined curvature (|∇f| = 0) are excluded from the average only for V2 and V3
        denom = mask_count if order == 1 else mask_count - nan_count
        return _MF_prefactor(self.dim, order) * tot / (dus * denom)

    def _run(self, fn, *args):
        """Call `fn(*args)`, raising any `AttributeError` as a `RuntimeError`. Otherwise, `pynkowski.V0`, ..., `pynkowski.V3` would catch it and silently use their loop over the thresholds instead."""
        try:
            return fn(*args)
        except AttributeError as e:
            raise RuntimeError(f"Internal error in {type(self).__name__} while computing a Minkowski Functional.") from e

    def _V0(self, us, dus, verbose=True):
        """Compute the first Minkowski Functional, $V_0$, normalized by the volume of the space. Internal interface for `pynkowski.V0`.

        Parameters
        ----------
        us : np.array
            The thresholds where $V_0$ is computed.

        dus : np.array
            The bin sizes of `us`. Ignored.

        verbose : bool, optional
            Ignored, as there is no progress bar for this computation.

        Returns
        -------
        V0 : np.array
            The values of the first Minkowski Functional at the given thresholds.

        """
        return self._run(self._v0, us)

    def _V1(self, us, dus, verbose=True):
        """Compute the second Minkowski Functional, $V_1$, normalized by the volume of the space. Internal interface for `pynkowski.V1`.

        Parameters
        ----------
        us : np.array
            The thresholds where $V_1$ is computed, as the centres of the bins.

        dus : np.array
            The bin sizes of `us`.

        verbose : bool, optional
            Ignored, as there is no progress bar for this computation.

        Returns
        -------
        V1 : np.array
            The values of the second Minkowski Functional at the given thresholds.

        """
        return self._run(self._mf, 1, us, dus)

    def _V2(self, us, dus, verbose=True):
        """Compute the third Minkowski Functional, $V_2$, normalized by the volume of the space. Internal interface for `pynkowski.V2`.

        Parameters
        ----------
        us : np.array
            The thresholds where $V_2$ is computed, as the centres of the bins.

        dus : np.array
            The bin sizes of `us`.

        verbose : bool, optional
            Ignored, as there is no progress bar for this computation.

        Returns
        -------
        V2 : np.array
            The values of the third Minkowski Functional at the given thresholds.

        """
        return self._run(self._mf, 2, us, dus)

    def _V3(self, us, dus, verbose=True):
        """Compute the fourth Minkowski Functional, $V_3$, normalized by the volume of the space. Internal interface for `pynkowski.V3`.

        Parameters
        ----------
        us : np.array
            The thresholds where $V_3$ is computed, as the centres of the bins.

        dus : np.array
            The bin sizes of `us`.

        verbose : bool, optional
            Ignored, as there is no progress bar for this computation.

        Returns
        -------
        V3 : np.array
            The values of the fourth Minkowski Functional at the given thresholds.

        """
        return self._run(self._mf, 3, us, dus)


__docformat__ = "numpy"
