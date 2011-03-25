#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: chnmf.py 20 2010-08-02 17:35:19Z cthurau $
#$Author$
"""      
PyMF Convex Hull Non-negative Matrix Factorization [1]
    
    CHNMF(NMF) : Class for Convex-hull NMF
    quickhull : Function for finding the convex hull in 2D
    
[1] C. Thurau, K. Kersting, and C. Bauckhage. Convex Non-Negative Matrix 
Factorization in the Wild. ICDM 2009.
"""

__version__ = "$Revision$"

import numpy as np

from itertools import combinations
from dist import vq
from pca import PCA
from aa import AA

__all__ = ["CHNMF"]


def quickhull(sample):
    """ Find data points on the convex hull of a supplied data set

    Args:
        sample: data points as column vectors n x d
                    n - number samples
                    d - data dimension (should be two)

    Returns:
        a k x d matrix containint the convex hull data points
    """

    link = lambda a, b: np.concatenate((a, b[1:]))
    edge = lambda a, b: np.concatenate(([a], [b]))

    def dome(sample, base): 
        h, t = base
        dists = np.dot(sample - h, np.dot(((0, -1), (1, 0)), (t - h)))
        outer = np.repeat(sample, dists > 0, axis=0)

        if len(outer):
            pivot = sample[np.argmax(dists)]
            return link(dome(outer, edge(h, pivot)),
                dome(outer, edge(pivot, t)))
        else:
            return base

    if len(sample) > 2:
        axis = sample[:, 0]
        base = np.take(sample, [np.argmin(axis), np.argmax(axis)], axis=0)
        return link(dome(sample, base),
            dome(sample, base[::-1]))
    else:
        return sample

class CHNMF(AA):
    """      
    CHNMF(data, num_bases=4, niter=100, show_progress=True, compute_w=True, compute_h=True)
        
    Convex Hull Non-negative Matrix Factorization. Factorize a data matrix into
    two matrices s.t. F = | data - W*H | is minimal. H is restricted to convexity 
    (H >=0, sum(H, axis=1) = [1 .. 1]) and W resides on actual data points. 
    Factorization is solved via an alternating least squares optimization using 
    the quadratic programming solver from cvxopt. The results are usually 
    equivalent to Archetypal Analysis (pymf.AA) but CHNMF also works for very 
    large datasets.
    
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
    base_sel: int,
        Number of pairwise basis vector projections. Set to a value< rank(data).
        Computation time scale exponentially with this value, usually rather low
        values are sufficient (3-10).
    
    Attributes
    ----------
        W : "data_dimension x num_bases" matrix of basis vectors
        H : "num bases x num_samples" matrix of coefficients
        ferr : frobenius norm (after calling .factorize()) 
    
    Example
    -------
    Applying CHNMF to some rather stupid data set:
    
    >>> import numpy as np
    >>> from chnmf import CHNMF
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    
    Use 2 basis vectors -> W shape(data_dimension, 2).    
    
    >>> chnmf_mdl = CHNMF(data, num_bases=2)
    
    And start computing the factorization.        
    
    >>> chnmf_mdl.factorize()
    
    The basis vectors are now stored in chnmf_mdl.W, the coefficients in 
    chnmf_mdl.H. To compute coefficients for an existing set of basis vectors 
    simply copy W to chnmf_mdl.W, and set compute_w to False:
    
    >>> data = np.array([[1.5, 2.0], [1.2, 1.8]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> chnmf_mdl = CHNMF(data, num_bases=2, niter=1, compute_w=False)
    >>> chnmf_mdl.W = W
    >>> chnmf_mdl.factorize()
    
    The result is a set of coefficients chnmf_mdl.H, s.t. data = W * chnmf_mdl.H.
    """        
    
    def __init__(self, data, num_bases=4, init_w=True, init_h=True, base_sel=3):
                             
        # call inherited method
        AA.__init__(self, data, num_bases=num_bases, 
                    init_w=init_w, init_h=init_h)
                
        # base sel should never be larger than the actual data dimension
        if base_sel < self.data.shape[0]:
            self._base_sel = base_sel
        else:
            self._base_sel = self.data.shape[0]

    def update_w(self): 
        """ compute new W """
        def select_hull_points(data, n=3):
            """ select data points for pairwise projections of the first n
            dimensions """
    
            # iterate over all projections and select data points
            idx = np.array([])

            # iterate over some pairwise combinations of dimensions
            for i in combinations(range(n), 2):
                # sample convex hull points in 2D projection                    
                convex_hull_d = quickhull(data[i, :].T)
            
                # get indices for convex hull data points
                idx = np.append(idx, vq(data[i, :], convex_hull_d.T))
                idx = np.unique(idx)
                
            return np.int32(idx)
    
        # determine convex hull data points only if the total
        # amount of available data is >50
        #if self.data.shape[1] > 50:    
        pcamodel = PCA(self.data)        
        pcamodel.factorize(show_progress=False)        
        self._hull_idx = select_hull_points(pcamodel.H, n=self._base_sel)

        #else:
        #    self._hull_idx = range(self.data.shape[1])

        aa_mdl = AA(self.data[:, self._hull_idx], num_bases=self._num_bases,                     
                    init_w=True, init_h=True)

        # determine W
        aa_mdl.factorize(niter=50, compute_h=True, compute_w=True, 
                         compute_err=True, show_progress=False)
            
        self.W = aa_mdl.W        
        self.map_W_to_Data()
        
        
    def factorize(self, niter=1, show_progress=False, 
                 compute_w=True, compute_h=True, compute_err=True):  
        """ Factorize s.t. WH = data
            
            Parameters
            ----------
            niter : int
                    number of iterations.
            show_progress : bool
                    print some extra information to stdout.
            compute_h : bool
                    iteratively update values for H.
            compute_w : bool
                    iteratively update values for W.
            compute_err : bool
                    compute Frobenius norm |data-WH| after each update and store
                    it to .ferr[k].
            
            Updated Values
            --------------
            .W : updated values for W.
            .H : updated values for H.
            .ferr : Frobenius norm |data-WH| for each iteration.
        """                     
        # set iterations to 1, otherwise it doesn't make sense
        AA.factorize(self, niter=1, show_progress=show_progress, 
                     compute_w=compute_w, compute_h=compute_h, 
                     compute_err=compute_err)      


if __name__ == "__main__":
    import doctest  
    doctest.testmod()    
