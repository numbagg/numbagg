from setuptools import find_packages, setup

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering",
]

DESCRIPTION = "Fast N-dimensional aggregation functions with Numba"

with open("README.md") as f:
    long_description = f.read()

setup(
    name="numbagg",
    license="BSD",
    author="Stephan Hoyer",
    author_email="shoyer@gmail.com",
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["numpy", "numba"],
    tests_require=["pytest", "bottleneck", "pandas"],
    setup_requires=["setuptools_scm"],
    python_requires=">=3.9",
    url="https://github.com/numbagg/numbagg",
    test_suite="pytest",
    packages=find_packages(),
    use_scm_version={"fallback_version": "999"},
)
