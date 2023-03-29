'''This module contains all the classes to define data fields in several conventions. So far, they are the following:

- [`scalar`](data/scalar.html): scalar maps on the sphere, in the healpix convention
- [`healpix`](data/healpix.html): scalar maps on the sphere in the healpix convention

- There is also a general utilities submodule called [`utils_da`](data/utils.html).
'''


# import numpy as np
# import healpy as hp

# try:
#     from tqdm.auto import tqdm
# except:
#     tqdm = lambda x: x
#     print('tqdm not loaded')

from .base_da import DataField

try:
    import healpy as hp
    from .healpix import Healpix, HealpixP2
    from .scalar import Scalar
except ImportError:
    hp = None
    print("healpy was not loaded, some functionality will be unavailable")
 

        
# __all__ = ["Healpix", "Scalar"]
__docformat__ = "numpy"
