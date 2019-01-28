from setuptools import setup, find_packages

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering",
]

DESCRIPTION = "Fast N-dimensional aggregation functions with Numba"

setup(
    name="numbagg",
    version="0.1",
    license="BSD",
    author="Stephan Hoyer",
    author_email="shoyer@gmail.com",
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    install_requires=["numpy", "numba"],
    tests_require=["pytest", "bottleneck", "pandas"],
    python_requires=">=3.5",
    url="https://github.com/shoyer/numbagg",
    test_suite="pytest",
    packages=find_packages(),
)
