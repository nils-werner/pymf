import setuptools

setuptools.setup(
    name='PyMF',
    version='0.1.9',
    description='Python Matrix Factorization Module',
    author='Christian Thurau',
    author_email='cthurau@googlemail.com',
    url='http://code.google.com/p/pymf/',
    packages=setuptools.find_packages(),
    license='OSI Approved :: GNU General Public License (GPL)',
    install_requires=[
        'cvxopt',
        'numpy',
        'scipy',
    ],
    extras_require={
        'tests': [
            'pytest',
            'pytest-cov',
        ],
    },
    tests_require=[
        'pytest',
        'pytest-cov',
    ],
    long_description=open('README.txt').read(),
)
