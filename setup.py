#!/usr/bin/env python
# encoding: utf-8


from setuptools import setup, find_packages

setup(
    name='dummy',
    version='0.1',
    description='dummy',
    author='Hervé Bredin',
    author_email='bredin@limsi.fr',
    url='http://packages.python.org/dummy',
    packages= find_packages(),
    install_requires=['numpy >=1.7.1',
                      'scipy >=0.12.0'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering"]
)
