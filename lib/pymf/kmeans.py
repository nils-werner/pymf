#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: kmeans.py 20 2010-08-02 17:35:19Z cthurau $
#$Author$
"""
PyMF K-means clustering (unary-convex matrix factorization).
"""

__version__ = "$Revision$"

import numpy as np
import random

import dist
from nmf import NMF

__all__ = ["Kmeans"]

class Kmeans(NMF):
    """      
    Kmeans(data, num_bases=4, niter=100, show_progress=True, compute_w=True)
    
    
    K-means clustering. Factorize a data matrix into two matrices s.t.
    F = | data - W*H | is minimal. H is restricted to unary vectors, W
    is simply the mean over the corresponding samples in data.
    
    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)    
    init_w: bool, optional
        Initialize W (True - default). Useful for using precomputed basis 
        vectors or custom initializations or matrices stored via hdf5.        
    init_h: bool, optional
        Initialize H (True - default). Useful for using precomputed coefficients 
        or custom initializations or matrices stored via hdf5.        
    

    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize()) 
    
    Example
    -------
    Applying K-means to some rather stupid data set:
    
    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> kmeans_mdl = Kmeans(data, num_bases=2, niter=10)
    >>> kmeans_mdl.initialization()
    >>> kmeans_mdl.factorize()
    
    The basis vectors are now stored in kmeans_mdl.W, the coefficients in kmeans_mdl.H. 
    To compute coefficients for an existing set of basis vectors simply    copy W 
    to kmeans_mdl.W, and set compute_w to False:
    
    >>> data = np.array([[1.5], [1.2]])
    >>> W = [[1.0, 0.0], [0.0, 1.0]]
    >>> kmeans_mdl = Kmeans(data, num_bases=2, niter=1, compute_w=False)
    >>> kmeans_mdl.initialization()
    >>> kmeans_mdl.W = W
    >>> kmeans_mdl.factorize()
    
    The result is a set of coefficients kmeans_mdl.H, s.t. data = W * kmeans_mdl.H.
    """
    
    def __init__(self, data, num_bases=4, init_w=True, init_h=True):
        
        # we don't need to initialize W,H in NMF
        NMF.__init__(self, data, num_bases=num_bases, 
                     init_w=False, init_h=False)
    
        # initialize W,H
        self.H = np.zeros((self._num_bases, self._num_samples))
        
        # set W to some random data samples
        sel = random.sample(xrange(self._num_samples), self._num_bases)
        
        # sort indices, otherwise h5py won't work
        self.W = self.data[:, np.sort(sel)]        
        self.update_h()
        

    def update_h(self):                    
        # and assign samples to the best matching centers
        self.assigned = dist.vq(self.W, self.data)
        self.H = np.zeros(self.H.shape)
        self.H[self.assigned, range(self._num_samples)] = 1.0
                
                    
    def update_w(self):            
        for i in range(self._num_bases):
            idx = np.where(self.assigned==i)[0]
            n = len(idx)        
            if n > 1:
                self.W[:,i] = np.sum(self.data[:,idx], axis=1)/n                    
