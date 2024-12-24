Installation
============

The Dual AutoDiff package is available in two versions:

1. ``dual_autodiff``: Pure Python implementation
2. ``dual_autodiff_x``: Cythonized version for improved performance

Using pip
---------

The easiest way to install the pure Python version is using pip::

    pip install dual_autodiff

For the optimized Cython version::

    pip install dual_autodiff_x

**(Sorry, I haven't uploaded the packages to PyPI yet. This section is just for demonstration.)**

From Source
-----------

To install from source, first clone the repository::

    git clone https://github.com/******/dual_autodiff.git
    cd dual_autodiff

Then install the package using pip::

    pip install .

For the Cythonized version::

    cd dual_autodiff_x
    pip install .

**(Sorry, I haven't uploaded the source code to GitHub yet. This section is just for demonstration.)**

Requirements
------------

Requirements:

Pure Python version (dual_autodiff):

* **Python 3.x**

Cythonized version (dual_autodiff_x):

* **Python 3.x+**
* **Cython**
* A C compiler (for building from source)

If you don't run the example notebooks, you don't need to install any additional dependencies.

Optional Dependencies
---------------------

For running example notebooks:

* **IPython**: For interactive computing
* **Jupyter**: For notebook interface

* **NumPy**: For numerical computations
* **Pandas**: For data manipulation and analysis
* **Matplotlib**: For plotting and visualization
* **seaborn**: For advanced plotting and visualization

Build Dependencies
------------------

For building the packages:

* **setuptools**
* **wheel**
* **Cython** (for dual_autodiff_x)

Development Dependencies
------------------------

For development and testing:

* **pytest**: For running tests
* **pytest-cov**: For test coverage reporting
* **sphinx**: For building documentation
* **nbsphinx**: For building documentation with Jupyter notebooks
* **sphinx-rtd-theme**: For building documentation with readthedocs theme

Verifying Installation
----------------------

To verify the pure Python version::

    >>> import dual_autodiff as df
    >>> x = df.Dual(2.0, 1.0)  # Should work without errors

For the Cythonized version::

    >>> import dual_autodiff_x as dfx
    >>> x = dfx.Dual(2.0, 1.0)  # Should work without errors