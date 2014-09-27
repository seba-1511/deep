Deep .. image:: https://travis-ci.org/GabrielPereyra/deep.svg?branch=master
    :target: https://travis-ci.org/GabrielPereyra/deep
==================

`Deep <http://deep.readthedocs.org>`_ provides a scikit-learn interface to
deep learning algorithms.


Supported Architectures
-----------------------

* `Tied Autoencoder <http://deep.readthedocs.org/en/latest/autoencoder.html>`_
* `Untied Autoencoder <http://deep.readthedocs.org/en/latest/autoencoder.html>`_
* `Denoising Autoencoder <http://deep.readthedocs.org/en/latest/autoencoder.html>`_

Init and Fit
------------

Here is a simple "Hello, world" example web app for Tornado::

    import deep.autoencoder.class.TiedAE
    import deep.datasets.base.load_mnist

    X = load_mnist()[0][0]
    ae = TiedAE(100)
    ae.fit(X)


This example initializes a tied wieght autoencoder with 100 hidden 
sigmoid units.

Installation
------------

**Automatic installation**::

    pip install deep

Tornado is listed in `PyPI <http://pypi.python.org/pypi/tornado/>`_ and
can be installed with ``pip`` or ``easy_install``.