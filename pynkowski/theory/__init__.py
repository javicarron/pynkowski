'''This module contains the theoretical predictions of the statistics for the implemented fields. So far, they are the following:

- [`base_th`](theory/base_th.html): The base class for theoretical fields, to be used as the base for the other fields.
- [`gaussian`](theory/gaussian.html): Isotropic Gaussian fields such as CMB temperature or initial density field.
- [`spingaussian`](theory/spingaussian.html): Isotropic Gaussian fields in the SO(3) formalism, such as the CMB polarization.
- [`chi2`](theory/chi2.html): Isotropic $\chi^2$ fields, such as the modulus of the CMB polarization.

- There is also a general utilities submodule called [`utils_th`](theory/utils_th.html).

These fields are defined with arbitrary dimensions and on an arbitrary space. Specific versions defined on particular spaces (such as the sphere) also exist for convenience.
'''

from .base_th import TheoryField
from .gaussian import SphericalGaussian, Gaussian #, EuclideanGaussian
from .spingaussian import SpinGaussian
from .chi2 import SphericalChi2, Chi2

__docformat__ = "numpy"
