from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("dual_autodiff_x.dual", ["dual_autodiff_x/dual.pyx"]),
]

setup(
    ext_modules = cythonize(extensions, compiler_directives={"language_level": "3"}),
    packages=["dual_autodiff_x"],

    # Include only .so/.pyd files (compiled extensions)
    package_data={"dual_autodiff_x": ["*.so", "*.pyd"]}, # .so for Linux, .pyd for Windows
    # Exclude source files to prevent the distribution of the source code
    exclude_package_data={"dual_autodiff_x": ["*.pyx", "*.py"]},
    # Ensure that wheels can be built
    zip_safe=False,
)