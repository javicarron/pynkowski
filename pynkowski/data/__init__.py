'''This module contains all the classes to work with data in several conventions. Currently, the implemented formats are the following:

- [`base_da`](data/base_da.html): the abstract class for general data fields, to be used as the base for the other fields.
- [`healpix`](data/healpix.html): scalar maps on the sphere in the healpix convention. An optional interface for $P^2$ is included.

- There is also a general utilities submodule called [`utils_da`](data/utils_da.html).
'''

from .base_da import DataField

try:
    import healpy as hp
    from .healpix import Healpix, HealpixP2
except ImportError:
    hp = None
    print("healpy was not loaded, some functionality will be unavailable")

# __all__ = ["Healpix", "HealpixP2"]
__docformat__ = "numpy"
