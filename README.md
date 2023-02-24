# Pynkowski

A Python package to compute Minkowski Functionals of input fields, as well as their expected values in the case of Gaussian isotropic fields.

The formats supported for **input data** are the following:

- Scalar HEALPix maps, as the ones used by [healpy](https://healpy.readthedocs.io/) (see [paper 1](https://arxiv.org/abs/2211.07562)).
- Polarisation HEALPix maps in the $SO(3)$ formalism (coming  soon, see [paper 2](https://arxiv.org/abs/2301.13191)).

...and more to come, feel free to contact us (by email [[1](mailto:javier.carron@roma2.infn.it) and [2](mailto:alessandro.carones@roma2.infn.it )] or opening an issue) to implement more schemes.


The **theoretical expectation** for different fields are implemented in the following cases:

- Gaussian scalar maps on the sphere (such as CMB $T$, see [paper 1](https://arxiv.org/abs/2211.07562)).
- $\chi^2$ maps on the sphere (such as CMB $P^2$, see [paper 1](https://arxiv.org/abs/2211.07562)).
- Spin 2 maps in the $SO(3)$ formalism (coming  soon, see [paper 2](https://arxiv.org/abs/2301.13191)).

...and more to come, feel free to contact us (by email [[1](mailto:javier.carron@roma2.infn.it) and [2](mailto:alessandro.carones@roma2.infn.it )] or opening an issue) to implement more theoretical expectations.

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
- [tqdm](https://github.com/tqdm/tqdm) (optional)

## Documentation

The documentation can be found on [https://javicarron.github.io/pynkowski](https://javicarron.github.io/pynkowski)

This package is divided into two modules: `data` and `theory`. Each module has a submodule for each kind of dataset or theoretical field, plus a general utilities submodule. In this way, extending the code to a new usecase is reduced to creating a new submodule. The structure is the following:

- [`data`](https://javicarron.github.io/pynkowski/pynkowski/data.html) 
    - [`scalar`](https://javicarron.github.io/pynkowski/pynkowski/data/scalar.html)
    - [`utils`](https://javicarron.github.io/pynkowski/pynkowski/data/utils.html)
    
- [`theory`](https://javicarron.github.io/pynkowski/pynkowski/theory.html)
    - [`temperature`](https://javicarron.github.io/pynkowski/pynkowski/theory/temperature.html)
    - [`p2`](https://javicarron.github.io/pynkowski/pynkowski/theory/p2.html)
    - [`utils`](https://javicarron.github.io/pynkowski/pynkowski/theory/utils.html)


## Example notebooks

- [Minkowski Functionals of a CMB temperature map and comparison with theory](https://github.com/javicarron/pynkowski/blob/main/examples/Temperature.ipynb).
- [Minkowski Functionals of a CMB polarization P² map and comparison with theory](https://github.com/javicarron/pynkowski/blob/main/examples/P2.ipynb).

## Authors

This package has been developed by [Javier Carrón Duque](https://www.javiercarron.com) and Alessandro Carones.
