'''This module contains all the classes to compute Minkowski Functionals on data. So far, they are the following:

- [`scalar`](data/scalar.html): scalar maps on the sphere, in the healpix convention,

- There is also a general utilities submodule called [`utils`](data/utils.html).
'''


import numpy as np
import healpy as hp

try:
    from tqdm.notebook import tqdm
except:
    tqdm = lambda x: x
    print('tqdm not loaded')
    
from .scalar import Scalar
 

        
#__all__ = ["Scalar"]
__docformat__ = "numpy"
