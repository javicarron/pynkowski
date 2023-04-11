'''This module contains all the implemented statistics that can be computed on data or simulations. They are divided into several submodules:

- [`minkowski`](stats/minkowski.html): Minkowski Functionals
- [`extrema`](stats/extrema.html): extrema distributions (maxima, minima),

- There is also a general utilities submodule called [`utils_st`](stats/utils_st.html).
'''

from .minkowski import V0, V1, V2, V3
from .extrema import maxima, minima
# from .utils_st import subsample_us, define_ubins


__docformat__ = "numpy"
