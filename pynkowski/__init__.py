'''A Python package to compute Minkowski Functionals,
as well as their Gaussian isotropic expected values
'''

from .data import Scalar

from .theory import (TheoryP2,
                     TheoryTemperature,
                     get_μ)

from .__version import __version__




__pdoc__ = {"version":False}
__docformat__ = "numpy"
