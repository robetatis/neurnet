from setuptools import find_packages, setup

setup(
    name='mnist',
    packages=['src'],
    version='0.0.1dev0',
    description='Feed forward dense neural network from scratch, with sigmoid activation, for solving MNIST classification',
    author='Dr.-Ing. Roberto Tatis-Muvdi',
    entry_points={'console_scripts': ['mnist=src.main:main']}
)
