#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: sivm.py 22 2010-08-13 11:16:43Z cthurau $
#$Author$
""" 
PyMF Simplex Volume Maximization for CUR [1]

    SIVM: class for SiVM

[1] C. Thurau, K. Kersting, and C. Bauckhage. Yes We Can - Simplex Volume 
Maximization for Descriptive Web-Scale Matrix Factorization. In Proc. Int. 
Conf. on Information and Knowledge Management. ACM. 2010.
"""

__version__ = "$Revision$"

import scipy.sparse
import numpy as np

from dist import *
from aa import AA

__all__ = ["SIVM"]

class SIVM(AA):
    """      
    SIVM(data, num_bases=4, niter=100, show_progress=True, compute_w=True)
    
    
    Simplex Volume Maximization. Factorize a data matrix into two matrices s.t.
    F = | data - W*H | is minimal. H is restricted to convexity. W is iteratively
    found by maximizing the volume of the resulting simplex (see [1]).
    
    Parameters
    ----------
    data : array_like [data_dimension x num_samples]
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
        Compute W (True) or only H (False). Useful for using basis vectors
        from another convexity constrained matrix factorization function
        (e.g. svmnmf) (if set to "True" niter can be set to "1")
    compute_h: bool, optional
        Compute H (True) or only H (False). Useful for using precomputed
        basis vectors.
    dist_measure: string, optional
        The distance measure for finding the next best candidate that 
        maximizes the simplex volume ['l2','l1','cosine','sparse_graph_l2']
        'l2' (default)
    optimize_lower_bound: bool, optional
        Use the alternative selection criterion that optimizes the lower
        bound (see [1])
        False (default)
    
    Attributes
    ----------
        W : "data_dimension x num_bases" matrix of basis vectors
        H : "num bases x num_samples" matrix of coefficients
        
        ferr : frobenius norm (after applying .factoriz())        
    
    Example
    -------
    Applying SIVM to some rather stupid data set:
    
    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> sivm_mdl = SIVM(data, num_bases=2, niter=10)
    >>> sivm_mdl.initialization()
    >>> sivm_mdl.factorize()
    
    The basis vectors are now stored in sivm_mdl.W, the coefficients in sivm_mdl.H. 
    To compute coefficients for an existing set of basis vectors simply    copy W 
    to sivm_mdl.W, and set compute_w to False:
    
    >>> data = np.array([[1.5, 1.3], [1.2, 0.3]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> sivm_mdl = SIVM(data, num_bases=2, niter=1, compute_w=False)
    >>> sivm_mdl.initialization()
    >>> sivm_mdl.W = W
    >>> sivm_mdl.factorize()
    
    The result is a set of coefficients sivm_mdl.H, s.t. data = W * sivm_mdl.H.
    """
    

    def __init__(self, data, num_bases=4, niter=1, 
                show_progress=False, compute_w=True, compute_h=True, 
                dist_measure='l2'):

        # call inherited method        
        # set "niter=1" as anything else doesn't make sense
        AA.__init__(self, data, num_bases=num_bases, niter=1, show_progress=show_progress, compute_w=compute_w, compute_h=compute_h)
            
        self._dist_measure = dist_measure            
        self._compute_h = compute_h        

        # assign the correct distance function
        if self._dist_measure == 'l1':
                self._distfunc = l1_distance
                
        elif self._dist_measure == 'l2':
                self._distfunc = l2_distance
        
        elif self._dist_measure == 'cosine':                
                self._distfunc = cosine_distance
        
        elif self._dist_measure == 'abs_cosine':                
                self._distfunc = abs_cosine_distance
                
        elif self._dist_measure == 'kl':
                self._distfunc = kl_divergence    
                
    def _distance(self, idx):
        # compute distances of a specific data point to all
        # other samples            
        if scipy.sparse.issparse(self.data):
            step = self.data.shape[1]
        else:    
            step = 50000    
                
        d = np.zeros((self.data.shape[1]))        
        if idx == -1:
            # set vec to origin if idx=-1
            vec = np.zeros((self.data.shape[0],1))
            if scipy.sparse.issparse(self.data):
                vec = scipy.sparse.csc_matrix(vec)
        else:
            vec = self.data[:, idx:idx+1]    
        
        self._logger.info('compute distance to node ' + str(idx))                                    
                                                
        # slice data into smaller chunks
        for idx_start in range(0, self.data.shape[1], step):                    
            if idx_start + step > self.data.shape[1]:
                idx_end = self.data.shape[1]
            else:
                idx_end = idx_start + step

            d[idx_start:idx_end] = self._distfunc(self.data[:,idx_start:idx_end], vec)
            self._logger.info('completed:' + str(idx_end/(self.data.shape[1]/100.0)) + "%")    
        return d
    
    def initialization(self, init='fastmap'):
        # initialization can be either 'fastmap' or 'origin' ...
        self.select = []

        if init == 'fastmap':
            # Fastmap like initialization
            # set the starting index for fastmap initialization        
            cur_p = 0        
            
            # after 3 iterations the first "real" index is found
            for i in range(3):                                
                d = self._distance(cur_p)        
                cur_p = np.argmax(d)
                
            # store maximal found distance -> later used for "a" (->update_w) 
            self._maxd = np.max(d)                        
            self.select.append(cur_p)

        elif init == 'origin':
            # set first vertex to origin
            cur_p = -1
            d = self._distance(cur_p)
            self._maxd = np.max(d)
            self.select.append(cur_p)
            

        if self._compute_h:
            self.H = np.zeros((self._num_bases, self._num_samples))
                
        self.W = np.zeros((self._data_dimension, self._num_bases))
        if scipy.sparse.issparse(self.data):
            self.W = scipy.sparse.csc_matrix(self.W)

    def update_w(self):        
        # initialize some of the recursively updated distance measures ....        
        EPS = 10**-8
        d_square = np.zeros((self.data.shape[1]))
        d_sum = np.zeros((self.data.shape[1]))
        d_i_times_d_j = np.zeros((self.data.shape[1]))
        distiter = np.zeros((self.data.shape[1]))

        a = np.log(self._maxd) 
        
        for l in range(1,self._num_bases):
            d = self._distance(self.select[-1])
            
            # take the log of d (sually more stable that d)
            d = np.log(d + EPS)            
            
            d_i_times_d_j += d * d_sum
            d_sum += d
            d_square += d**2
            distiter = d_i_times_d_j + a*d_sum - (l/2.0) * d_square        

            # detect the next best data point
            self._logger.info('searching for next best node ...')                    
            self.select.append(np.argmax(distiter))
            self._logger.info('cur_nodes: ' + str(self.select))

        # sort indices, otherwise h5py won't work
        self.W = self.data[:, np.sort(self.select)]
            
        # "unsort" it again to keep the correct order
        self.W = self.W[:, np.argsort(np.argsort(self.select))]
                    
if __name__ == "__main__":
    import doctest  
    doctest.testmod()    
