'''# Welcome to Pynkowski's documentation!

A Python package to compute Minkowski Functionals of input fields, as well as their expected values in the case of Gaussian isotropic fields.

The formats supported for **input data** are the following:
- Scalar HEALPix maps, as the ones used by [healpy](https://healpy.readthedocs.io/) (see paper).

...and more to come, feel free to contact us (by [email](mailto:javier.carron@roma2.infn.it) or opening an issue) to implement more schemes.


The **theoretical expectation** for Gaussian isotropic fields are implemented in the following cases:
- Gaussian scalar maps on the sphere (such as CMB $T$, see paper).
- $\chi^2$ maps on the sphere (such as CMB $P^2$, see paper).

...and more to come, feel free to contact us (by [email](mailto:javier.carron@roma2.infn.it) or opening an issue) to implement more theoretical expectations.

The repository can be found on [https://github.com/javicarron/pynkowski](https://github.com/javicarron/pynkowski).

## Installation

This package can be installed with: 
```
pip install pynkowski
```

The dependencies are:
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [healpy](https://healpy.readthedocs.io/)
- [tqdm](https://github.com/tqdm/tqdm) (optional, notebook only)

## Documentation

The documentation can be found on [https://javicarron.github.io/pynkowski](https://javicarron.github.io/pynkowski)

## Example notebooks

- Get the Minkowski Functionals of a CMB temperature $T$ map and compare with theory.
- Get the Minkowski Functionals of a CMB polarization $P^2=Q^2+U^2$ map and compare with theory.


## Authors

This package has been developed by [Javier Carrón Duque](https:www.javiercarron.com) and Alessandro Carones.
'''

from .data import Scalar

from .theory import (get_μ,
                     TheoryTemperature,
                     TheoryP2)

from .__version import __version__




__docformat__ = "numpy"
