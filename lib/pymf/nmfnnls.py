#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: kmeans.py 20 2010-08-02 17:35:19Z cthurau $
#$Author$
"""
PyMF Non-negative Matrix Factorization.

    NMFALS: Class for Non-negative Matrix Factorization using non negative 
            least squares optimization (requires scipy.optimize)

[1] Lee, D. D. and Seung, H. S. (1999), Learning the Parts of Objects by Non-negative 
Matrix Factorization, Nature 401(6755), 788-799.
"""

__version__ = "$Revision$"


import scipy.optimize
from nmf import NMF

__all__ = ["NMFNNLS"]

class NMFNNLS(NMF):
    """      
    NMFNNLS(data, num_bases=4, niter=100, show_progress=False, compute_w=True)
    
    
    Non-negative Matrix Factorization. Factorize a data matrix into two matrices 
    s.t. F = | data - W*H | = | is minimal. H, and W are restricted to non-negative
    data. Uses the Lawsons and Hanson's algorithm for non negative constrained
    least squares (-> also see scipy.optimize.nnls)
    
    Parameters
    ----------
    data : array_like
        the input data
    num_bases: int, optional 
        Number of bases to compute (column rank of W and row rank of H). 
        4 (default)
    niter: int, optional
        Number of iterations of the alternating optimization.
        100 (default)
    show_progress: bool, optional
        Print some extra information
        False (default)
    compute_w: bool, optional
        Compute W (True) or only H (False). Useful for using precomputed
        basis vectors.
    
    Attributes
    ----------
        W : "data_dimension x num_bases" matrix of basis vectors
        H : "num bases x num_samples" matrix of coefficients    
    
    Example
    -------
    Applying NMF to some rather stupid data set:
    
    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> nmf_mdl = NMFALS(data, num_bases=2, niter=10)
    >>> nmf_mdl.initialization()
    >>> nmf_mdl.factorize()
    
    The basis vectors are now stored in nmf_mdl.W, the coefficients in nmf_mdl.H. 
    To compute coefficients for an existing set of basis vectors simply    copy W 
    to nmf_mdl.W, and set compute_w to False:
    
    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> nmf_mdl = NMFALS(data, num_bases=2, niter=1, compute_w=False)
    >>> nmf_mdl.initialization()
    >>> nmf_mdl.W = W
    >>> nmf_mdl.factorize()
    
    The result is a set of coefficients nmf_mdl.H, s.t. data = W * nmf_mdl.H.
    """
    
    def __init__(self, data, num_bases=4, niter=10, show_progress=False, compute_w=True):
        
        NMF.__init__(self, data, num_bases=num_bases, niter=niter, show_progress=show_progress, compute_w=compute_w)


    def update_h(self):
        def updatesingleH(i):        
            self.H[:,i] = scipy.optimize.nnls(self.W, self.data[:,i])[0]
                                                                            
        map(updatesingleH, xrange(self._num_samples))                        
            
                
    def update_w(self):
        def updatesingleW(i):            
            self.W[i,:] = scipy.optimize.nnls(self.H.T, self.data[i,:].T)[0]
                
        map(updatesingleW, xrange(self._data_dimension))
