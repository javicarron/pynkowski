'''This module contains all the implemented statistics that can be computed on data or simulations. They are divided into several submodules:

- [`minkowski`](stats/minkowski.html): Minkowski Functionals
- [`extrema`](stats/extrema.html): extrema distributions (maxima, minima),

- There is also a general utilities submodule called [`utils`](stats/utils.html).
'''


# import numpy as np
# import healpy as hp

try:
    from tqdm.auto import tqdm
except:
    tqdm = lambda x: x
    print('tqdm not loaded')
    
# from .scalar import Scalar
 

        
#__all__ = ["Scalar"]
__docformat__ = "numpy"
