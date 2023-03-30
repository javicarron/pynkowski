'''This module contains all the theoretical predictions for the implemented fields. So far, they are the following:

- [`temperature`](theory/temperature.html): Gaussian fields such as CMB temperature.
- [`p2`](theory/p2.html): $\chi^2$ fields, such as the modulus of the CMB polarization.

- There is also a general utilities submodule called [`utils`](theory/utils.html).
'''


# import numpy as np
# import scipy.stats

# norm = scipy.stats.norm()

# from .utils_th import get_μ, define_mu, subsample_us   #define_us_for_V
from .base_th import TheoryField
from .gaussian import SphericalGaussian, Gaussian
from .chi2 import SphericalChi2, Chi2



#__all__ = ["get_μ",
           #"TheoryTemperature",
           #"TheoryP2"]

__docformat__ = "numpy"
