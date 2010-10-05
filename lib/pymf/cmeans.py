#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: kmeans.py 20 2010-08-02 17:35:19Z cthurau $
#$Author$
"""
PyMF K-means clustering (unary-convex matrix factorization).
Copyright (C) Christian Thurau, 2010. GNU General Public License (GPL). 
"""

__version__ = "$Revision$"


import time
import numpy as np
import random

import dist
from nmf import NMF

__all__ = ["Cmeans"]

class Cmeans(NMF):
	"""  	
	cmeans(data, num_bases=4, niter=100, show_progress=True, compW=True)
	
	
	Fuzzy c-means soft clustering. Factorize a data matrix into two matrices s.t.
	F = | data - W*H | is minimal. H is restricted to convexity (columns
	sum to 1) W	is simply the weighted mean over the corresponding samples in 
	data. Note that the objective function is based on distances (?), hence the
	Frobenius norm is probably not a good quality measure.
	
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
		Compute W (True) or only H (False). Useful for using precomputed
		basis vectors.
	
	Attributes
	----------
		W : "data_dimension x num_bases" matrix of basis vectors
		H : "num bases x num_samples" matrix of coefficients
		
		ferr : frobenius norm (after calling .factorize())
	
	Example
	-------
	Applying C-means to some rather stupid data set:
	
	>>> import numpy as np
	>>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
	>>> cmeans_mdl = Cmeans(data, num_bases=2, niter=10)
	>>> cmeans_mdl.initialization()
	>>> cmeans_mdl.factorize()
	
	The basis vectors are now stored in cmeans_mdl.W, the coefficients in cmeans_mdl.H. 
	To compute coefficients for an existing set of basis vectors simply	copy W 
	to cmeans_mdl.W, and set compW to False:
	
	>>> data = np.array([[1.5], [1.2]])
	>>> W = [[1.0, 0.0], [0.0, 1.0]]
	>>> cmeans_mdl = Cmeans(data, num_bases=2, niter=1, compW=False)
	>>> cmeans_mdl.initialization()
	>>> cmeans_mdl.W = W
	>>> cmeans_mdl.factorize()
	
	The result is a set of coefficients kmeans_mdl.H, s.t. data = W * kmeans_mdl.H.
	"""
	
	_VINFO = 'pymf-kmeans v0.1'
	_EPS = 10**-8
	
	def __init__(self, data, num_bases=4, niter=50, show_progress=False, compW=True):
		
		NMF.__init__(self, data, num_bases=num_bases, niter=niter, show_progress=show_progress, compW=compW)


	def initialization(self):
		# initialize W,H
		self.H = np.random.random((self._num_bases, self._num_samples))
		
		# set W to some random data samples
		self.H = self.H / np.sum(self.H, axis=0)
		
		# sort indices, otherwise h5py won't work
		self.W = np.zeros((self._data_dimension, self._num_bases))
		self.updateW()
		

	def updateH(self):					
		# assign samples to best matching centres ...
		m = 1.75
		tmp_dist = dist.pdist(self.W, self.data, metric='l2') + self._EPS
		self.H[:,:] = 0.0
		
		for i in range(self._num_bases):
			for k in range(self._num_bases):				
					self.H[i,:] += (tmp_dist[i,:]/tmp_dist[k,:])**(2.0/(m-1))
			
		self.H = np.where(self.H>0, 1.0/self.H, 0)	
					
	def updateW(self):			
		for i in range(self._num_bases):
			tmp = (self.H[i:i+1,:] * self.data).sum(axis=1)
			self.W[:,i] = tmp/(self.H[i,:].sum() + self._EPS)		