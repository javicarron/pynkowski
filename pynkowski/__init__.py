'''# Welcome to Pynkowski's documentation!

A Python package to compute Minkowski Functionals and other higher order statistics of input fields, as well as their expected values for different kinds of fields.

The **statistics** currently supported by this package are:

- Minkowski functionals.
- Maxima and minima distributions.

The formats currently supported for **input data** are the following:

- Scalar HEALPix maps, as the ones used by [healpy](https://healpy.readthedocs.io/), such as $T, \kappa, P^2$ (see [paper 1](https://arxiv.org/abs/2211.07562)).
- Polarisation HEALPix maps in the $SO(3)$ formalism (see [paper 2](https://arxiv.org/abs/2301.13191)).
- 2D and 3D numpy arrays (coming soon).

The theoretical expectation of some statistics is currently supported for the following **theoretical fields**:

- Gaussian fields (such as CMB $T$ or the initial density field, see [paper 1](https://arxiv.org/abs/2211.07562)).
- $\chi^2$ fields (such as CMB $P^2$, see [paper 1](https://arxiv.org/abs/2211.07562)).
- Spin 2 maps in the $SO(3)$ formalism (see [paper 2](https://arxiv.org/abs/2301.13191)).

We are actively working on the implementation of more statistics, data formats, and theoretical fields. If you want to contribute, we welcome and appreciate pull requests. 
If you have any comments or suggestions, please feel free to contact us by email ([1](mailto:javier.carron@csic.es) and [2](mailto:alessandro.carones@roma2.infn.it )) or by opening a discussion thread or issue.

The repository can be found on [https://github.com/javicarron/pynkowski](https://github.com/javicarron/pynkowski).

# Installation

This package can be installed with: 
```
pip install pynkowski
```

The dependencies are:

- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [healpy](https://healpy.readthedocs.io/)
- [tqdm](https://github.com/tqdm/tqdm) (optional)

# Documentation

The documentation can be found on [https://javicarron.github.io/pynkowski](https://javicarron.github.io/pynkowski).

This package is divided into three modules: `stats`, `data`, and `theory`. Each module has a submodule for each kind of object, plus a general utilities submodule and a base submodule for the definition of the base class. In this way, extending the code to a new usecase is reduced to creating a new submodule. The structure is the following:

- [`stats`](https://javicarron.github.io/pynkowski/pynkowski/stats.html)
    - [`minkowski`](https://javicarron.github.io/pynkowski/pynkowski/stats/minkowski.html)
    - [`extrema`](https://javicarron.github.io/pynkowski/pynkowski/stats/extrema.html)
    - [`utils_st`](https://javicarron.github.io/pynkowski/pynkowski/stats/utils_st.html)

- [`data`](https://javicarron.github.io/pynkowski/pynkowski/data.html)
    - [`base_da`](https://javicarron.github.io/pynkowski/pynkowski/data/base_da.html)
    - [`array`](https://javicarron.github.io/pynkowski/pynkowski/data/array.html)
    - [`healpix`](https://javicarron.github.io/pynkowski/pynkowski/data/healpix.html)
    - [`utils_da`](https://javicarron.github.io/pynkowski/pynkowski/data/utils_da.html)
  
- [`theory`](https://javicarron.github.io/pynkowski/pynkowski/theory.html)
    - [`base_th`](https://javicarron.github.io/pynkowski/pynkowski/theory/base_th.html)
    - [`gaussian`](https://javicarron.github.io/pynkowski/pynkowski/theory/gaussian.html)
    - [`spingaussian`](https://javicarron.github.io/pynkowski/pynkowski/theory/spingaussian.html)
    - [`chi2`](https://javicarron.github.io/pynkowski/pynkowski/theory/chi2.html)
    - [`utils_th`](https://javicarron.github.io/pynkowski/pynkowski/theory/utils_th.html)

The documentation for each submodule can be found by clicking on the links above or navigating the menu on the left.

# Example notebooks

- [Minkowski Functionals of a CMB temperature map and comparison with theory](https://github.com/javicarron/pynkowski/blob/main/examples/Temperature.ipynb).
- [Minkowski Functionals of a CMB polarization P² map and comparison with theory](https://github.com/javicarron/pynkowski/blob/main/examples/P2.ipynb).

# Authors

This package has been developed by [Javier Carrón Duque](https://www.javiercarron.com) and Alessandro Carones.

'''

from .data import (DataField, 
                   Healpix,
                   HealpixP2,
                   DataArray,
                   SO3Healpix,
                   SO3Patch)

from .theory import (TheoryField,
                     Gaussian,
                     SphericalGaussian,
                    #  EuclideanGaussian,
                     SpinGaussian,
                     Chi2,
                     SphericalChi2)

from .stats import (V0, V1, V2, V3, maxima, minima)

from .__version import __version__




__docformat__ = "numpy"
