#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: sivm.py 22 2010-08-13 11:16:43Z cthurau $
#$Author$
""" 
PyMF LAESA
"""

__version__ = "$Revision$"

import scipy.sparse
import numpy as np

from dist import *
from aa import AA

__all__ = ["LAESA"]

class LAESA(AA):
    """      
    SIVM(data, num_bases=4, niter=100, show_progress=True, compute_w=True)
    
    
    Simplex Volume Maximization. Factorize a data matrix into two matrices s.t.
    F = | data - W*H | is minimal. H is restricted to convexity. W is iteratively
    found by maximizing the volume of the resulting simplex (see [1]).
    
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
    Applying LAESA to some rather stupid data set:
    
    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> laesa_mdl = LAESA(data, num_bases=2, niter=10)
    >>> laesa_mdl.initialization()
    >>> laesa_mdl.factorize()
    
    The basis vectors are now stored in laesa_mdl.W, the coefficients in laesa_mdl.H. 
    To compute coefficients for an existing set of basis vectors simply    copy W 
    to laesa_mdl.W, and set compute_w to False:
    
    >>> data = np.array([[1.5, 1.3], [1.2, 0.3]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> laesa_mdl = LAESA(data, num_bases=2, niter=1, compute_w=False)
    >>> laesa_mdl.initialization()
    >>> laesa_mdl.W = W
    >>> laesa_mdl.factorize()
    
    The result is a set of coefficients laesa_mdl.H, s.t. data = W * laesa_mdl.H.
    """

    def __init__(self, data, num_bases=4, init_w=True, init_h=True, dist_measure='l2'):

        # call inherited method        
        AA.__init__(self, data, num_bases=num_bases, init_w=init_w, init_h=init_h)
            
        self._dist_measure = dist_measure    

        # assign the correct distance function
        if self._dist_measure == 'l1':
                self._distfunc = l1_distance
                
        elif self._dist_measure == 'l2':
                self._distfunc = l2_distance
        elif self._dist_measure == 'cosine':
                self._distfunc = cosine_distance
                
        elif self._dist_measure == 'kl':
                self._distfunc = kl_divergence    
                        
        elif self._dist_measure == 'sparse_graph_l2':
                self._distfunc = sparse_graph_l2_distance

                       

    def _distance(self, idx):
            # compute distances of a specific data point to all
            # other samples            
            if scipy.sparse.issparse(self.data):
                step = self.data.shape[1]
            else:    
                step = 50000    
                
            d = np.zeros((self.data.shape[1]))        
            vec = self.data[:, idx:idx+1]    
            self._logger.info('compute distance to node ' + str(idx))                                    
            # slice data into smaller chunks
            for idx_start in range(0,self.data.shape[1],step):                    
                if idx_start + step > self.data.shape[1]:
                    idx_end = self.data.shape[1]
                else:
                    idx_end = idx_start + step

                d[idx_start:idx_end] = self._distfunc(self.data[:,idx_start:idx_end], vec)
                self._logger.info('completed:' + str(idx_end/(self.data.shape[1]/100.0)) + "%")    
            
            return d

    def update_w(self):    
             
        
        
        # initialize some of the recursively updated distance measures       
        distiter = self._distance(self.select[-1])                
        
        for l in range(self._num_bases-1):                                        
            d = self._distance(self.select[-1])                                
        
            # replace distances in distiter
            distiter = np.where(d<distiter, d, distiter)
            
            # detect the next best data point
            self._logger.info('searching for next best node ...')                    
            self.select.append(np.argmax(distiter))
            self._logger.info('cur_nodes: ' + str(self.select))

        # sort indices, otherwise h5py won't work
        self.W = self.data[:, np.sort(self.select)]
        # but "unsort" it again to keep the correct order
        self.W = self.W[:, np.argsort(np.argsort(self.select))]    
                   
    
    def factorize(self, niter=1, show_progress=False, 
                  compute_w=True, compute_h=True, compute_err=True): 
        """ Factorize s.t. WH = data
            
            Parameters
            ----------           
            show_progress : bool
                    print some extra information to stdout.
            compute_h : bool
                    iteratively update values for H.
            compute_w : bool
                    iteratively update values for W.
            compute_err : bool
                    compute Frobenius norm |data-WH| and store it to .ferr
            
            Updated Values
            --------------
            .W : updated values for W.
            .H : updated values for H.
            .ferr : Frobenius norm |data-WH|
        """         
        # Fastmap like initialization
        # set the starting index for fastmap initialization        
        cur_p = 0        
        # after 3 iterations the first "real" index is found
        for i in range(3):                                
            d = self._distance(cur_p)        
            cur_p = np.argmax(d)

        self.select = []
        self.select.append(cur_p)                
                      
        # set iterations to 1, otherwise it doesn't make sense
        AA.factorize(self, niter=1, show_progress=show_progress, 
                     compute_w=compute_w, compute_h=compute_h, 
                     compute_err=compute_err)                  
                   
if __name__ == "__main__":
    import doctest  
    doctest.testmod()    
