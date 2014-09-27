Deep 
==================
.. image:: https://travis-ci.org/GabrielPereyra/deep.svg?branch=master :target: https://travis-ci.org/GabrielPereyra/deep
.. image:: https://coveralls.io/repos/GabrielPereyra/deep/badge.png :target: https://coveralls.io/r/GabrielPereyra/deep

`Deep <http://deep.readthedocs.org>`_ provides a scikit-learn interface to
deep learning algorithms.


Supported Architectures
-----------------------

* `Tied Autoencoder <http://deep.readthedocs.org/en/latest/autoencoder.html>`_
* `Untied Autoencoder <http://deep.readthedocs.org/en/latest/autoencoder.html>`_
* `Denoising Autoencoder <http://deep.readthedocs.org/en/latest/autoencoder.html>`_

Init and Fit
------------

Here is a simple example of fitting an autoencoder on MNIST::

    from deep.autoencoder import TiedAE
    from deep.datasets import load_mnist

    X = load_mnist()[0][0]
    ae = TiedAE(100)
    ae.fit(X)

This example initializes a tied weight autoencoder with 100 hidden 
sigmoid units.

Installation
------------

**Automatic installation**::

    pip install deep

Deep is listed in `PyPI <http://pypi.python.org/pypi/deep/>`_ and
can be installed with ``pip`` or ``easy_install``.