'''This module contains all the implemented statistics that can be computed on data or simulations. They are divided into several submodules:

- [`minkowski`](stats/minkowski.html): Minkowski Functionals
- [`extrema`](stats/extrema.html): extrema distributions (maxima, minima),

- There is also a general utilities submodule called [`utils`](stats/utils.html).
'''

from .minkowski import V0, V1, V2
from .utils_st import subsample_us, define_ubins

        
#__all__ = ["Scalar"]
__docformat__ = "numpy"
