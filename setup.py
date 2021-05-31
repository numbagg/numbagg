from setuptools import find_packages, setup

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering",
]

DESCRIPTION = "Fast N-dimensional aggregation functions with Numba"

setup(
    name="numbagg",
    version="0.2.0",
    license="BSD",
    author="Stephan Hoyer",
    author_email="shoyer@gmail.com",
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    install_requires=["numpy", "numba"],
    tests_require=["pytest", "bottleneck", "pandas"],
    python_requires=">=3.7",
    url="https://github.com/numbagg/numbagg",
    test_suite="pytest",
    packages=find_packages(),
)
