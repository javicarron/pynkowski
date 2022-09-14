 
from setuptools import setup, find_namespace_packages
from pynkowski.__version import __version__


setup(
    name='pynkowski',
    version=__version__,
    description='A Python package to compute Minkowski Functionals of input fields, as well as their expected values in the case of Gaussian isotropic fields.',
    license='GNU General Public License v3.0',
    license_files=['LICENSE'],
    author="Carrón Duque, Javier and Carones, Alessando",
    author_email='javier.carron@roma2.infn.it',
    packages=['pynkowski','pynkowski.data','pynkowski.theory'],
    package_dir={"pynkowski": "pynkowski"},
    url='https://github.com/javicarron/pynkowski',
    keywords='minkowski-functionals,non-gaussian,spherical,anisotropy,healpy,polarization,minkowski,cmb,healpix,curvature,cosmology,gaussian,maps,topology',
    install_requires=[
          'numpy',
          'scipy',
          'healpy'
      ],

)
