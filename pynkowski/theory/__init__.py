'''This module contains all the theoretical predictions for the implemented fields. So far, they are the following:

- [`temperature`](theory/temperature.html): Gaussian fields such as CMB temperature.
- [`p2`](theory/p2.html): $\chi^2$ fields, such as the modulus of the CMB polarization.

- There is also a general utilities submodule called [`utils`](theory/utils.html).
'''


import numpy as np
import scipy.stats

norm = scipy.stats.norm()

from .utils import get_μ, define_mu, define_us_for_V
from .temperature import TheoryTemperature
from .p2 import TheoryP2


#__all__ = ["get_μ",
           #"TheoryTemperature",
           #"TheoryP2"]

__docformat__ = "numpy"
