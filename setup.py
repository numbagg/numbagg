from setuptools import setup, find_packages

setup(name='numbagg',
      version='0.1-dev',
      license='MIT',
      author='Stephan Hoyer',
      author_email='shoyer@gmail.com',
      install_requires=['numpy', 'numba'],
      tests_require=['nose'],
      url='https://github.com/shoyer/numbagg',
      test_suite='nose.collector',
      packages=find_packages())
