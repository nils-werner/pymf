import os
from distutils.core import setup

setup(name='PyMF',
      version='0.1',
      description='Python Matrix Factorization Library',
      author='Christian Thurau',
      author_email='cthurau@gmail.com',
      url='http://cthurau.fargonauten.de',
      package_dir = {'': 'lib'},
      packages = ["pymf"],
      scripts=['scripts/pymftest.py'],
      license = 'OSI Approved :: GNU General Public License (GPL)',
      install_requires=[
        'cvxopt',
        'pygame (>=1.0)']      
      )     