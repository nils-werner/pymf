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
	
[1] C. Thurau, K. Kersting, and C. Bauckhage. Convex Non-Negative Matrix Factorization 
in the Wild. ICDM 2009.
"""

__version__ = "$Revision$"

import numpy as np

#from itertools import combinations

from dist import vq
from pca import PCA
from aa import AA

__all__ = ["CHNMF"]

def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


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
	CHNMF(data, num_bases=4, niter=100, show_progress=True, compW=True, compH=True)
		
	Convex Hull Non-negative Matrix Factorization. Factorize a data matrix into two 
	matrices s.t. F = | data - W*H | is minimal. H is restricted to convexity 
	(H >=0, sum(H, axis=1) = [1 .. 1]) and W resides on actual data points. 
	Factorization is solved via an alternating least squares optimization using 
	the quadratic programming solver from cvxopt. The results are usually equivalent 
	to Archetypal Analysis (pymf.AA) but CHNMF also works for very large datasets.
	
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
	compW: bool, optional
		Compute W (True) or only H (False). Useful for using basis vectors
		from another convexity constrained matrix factorization function
		(e.g. svmnmf) (if set to "True" niter can be set to "1")
	compH: bool, optional
		Compute H (True) or only W (False). Useful when only the basis vectors
		need to be known.
	
	Attributes
	----------
		W : "data_dimension x num_bases" matrix of basis vectors
		H : "num bases x num_samples" matrix of coefficients
	
		ferr : frobenius norm (after calling .factorize())
	
	Example
	-------
	Applying CHNMF to some rather stupid data set:
	
	>>> import numpy as np
	>>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
	>>> chnmf_mdl = CHNMF(data, num_bases=2, niter=10)
	>>> chnmf_mdl.initialization()
	>>> chnmf_mdl.factorize()
	
	The basis vectors are now stored in chnmf_mdl.W, the coefficients in chnmf_mdl.H. 
	To compute coefficients for an existing set of basis vectors simply	copy W 
	to chnmf_mdl.W, and set compW to False:
	
	>>> data = np.array([[1.5, 2.0], [1.2, 1.8]])
	>>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
	>>> chnmf_mdl = CHNMF(data, num_bases=2, niter=1, compW=False)
	>>> chnmf_mdl.initialization()
	>>> chnmf_mdl.W = W
	>>> chnmf_mdl.factorize()
	
	The result is a set of coefficients chnmf_mdl.H, s.t. data = W * chnmf_mdl.H.
	"""		
	
	def __init__(self, data, num_bases=4,  niter=100, show_progress=False, compW=True, compH=True, base_sel=3):		
		# call inherited method
		AA.__init__(self, data, num_bases=num_bases, niter=niter, show_progress=show_progress, compW=compW)
				
		self._compH = compH
				
		# base sel should never be larger than the actual
		# data dimension
		if base_sel < self.data.shape[0]:
			self._base_sel = base_sel
		else:
			self._base_sel = self.data.shape[0]

	def updateW(self):
		
		def selectHullPoints(data, n=3):
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
		pcamodel = PCA(self.data, show_progress=self._show_progress)		
		pcamodel.factorize()		
		self._hull_idx = selectHullPoints(pcamodel.H, n=self._base_sel)

		#else:
		#	self._hull_idx = range(self.data.shape[1])

		aa_mdl = AA(self.data[:, self._hull_idx], num_bases=self._num_bases , niter=self._niter, show_progress=self._show_progress, compW=True)

		# initialize W, H, and beta
		aa_mdl.initialization()

		# determine W
		aa_mdl.factorize()
			
		self.W = aa_mdl.W		
	
	def factorize(self):
		"""Do factorization s.t. data = dot(dot(data,beta),H), under the convexity constraint
			beta >=0, sum(beta)=1, H >=0, sum(H)=1
		"""
		# compute new coefficients for reconstructing data points
		if self._compW:
			self.updateW()
			self.map_W_to_Data()
		
		# for CHNMF it is sometimes useful to only compute
		# the basis vectors
		if self._compH:
			self.updateH()
					
		self.ferr = np.zeros(1)
		self.ferr[0] = self.frobenius_norm()		
		self._logger.info('FN:' + str(self.ferr[0]))	

if __name__ == "__main__":
	import doctest  
	doctest.testmod()	
