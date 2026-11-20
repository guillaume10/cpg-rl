from setuptools import find_packages
from distutils.core import setup

setup(
    name='cpg_gym',
    version='1.0.0',
    author='Guillaume Bellegarda',
    license="BSD-3-Clause",
    packages=find_packages(),
    description='Learning CPGs for terrain-aware legged robot skills',
    install_requires=['isaacgym',
                      'rsl-rl',
                      'matplotlib']
)