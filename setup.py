import os

from setuptools import setup

def read(*paths):
    with open(os.path.join(*paths), 'r') as f:
        return f.read()

def setup_package():
    metadata = dict(name='deep',
                    version='0.0.1',
                    url='http://github.com/gabrielpereyra/deep/',
                    license='BSD',
                    author='Gabriel Pereyra',
                    author_email='gbrl.pereyra@gmail.com',
                    description='Scikit-learn interface for deep-learning',
                    long_description='',
                    intall_requires=[
                        'numpy==1.8.1',
                        'scikit-learn==0.15.2',
                        'scipy==0.14.0',
                        'Theano==0.6.0',
                    ],
                    classifiers=['Intended Audience :: Science/Research',
                                 'Intended Audience :: Developers',
                                 'License :: OSI Approved',
                                 'Programming Language :: Python',
                                 'Topic :: Software Development',
                                 'Topic :: Scientific/Engineering',
                                 'Operating System :: Microsoft :: Windows',
                                 'Operating System :: POSIX',
                                 'Operating System :: Unix',
                                 'Operating System :: MacOS',
                                 'Programming Language :: Python :: 2',
                                 'Programming Language :: Python :: 2.6',
                                 'Programming Language :: Python :: 2.7',
                                 'Programming Language :: Python :: 3',
                                 'Programming Language :: Python :: 3.3',
                                 'Programming Language :: Python :: 3.4',
                                 ])

if __name__ == "__main__":
    setup_package()
