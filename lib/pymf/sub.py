#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: sub.py 20 2010-08-02 17:35:19Z cthurau $
#$Author$
"""  
PyMF Matrix SubSampling.
Copyright (C) Christian Thurau, 2010. GNU General Public License (GPL). 
"""

__version__ = "$Revision$"


import numpy as np
import random
from itertools import combinations

import dist
from chnmf import quickhull
from nmf import *
from pca import *
from kmeans import *

__all__ = ["SUB"]

class SUB(NMF):
	"""  	
	aa(data, num_bases=4, niter=100, show_progress=True, compW=True)
	
	
	Archetypal Analysis. Factorize a data matrix into two matrices s.t.
	F = | data - W*H | = | data - data*beta*H| is minimal. H and beta
	are restricted to convexity (beta >=0, sum(beta, axis=1) = [1 .. 1]).
	Factorization is solved via an alternating least squares optimization
	using the quadratic programming solver from cvxopt.
	
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
	
	Attributes
	----------
	
	Example
	-------
	>>> import numpy as np
	>>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
	>>> aa_mdl = AA(data, num_bases=2, niter=10)
	>>> aa_mdl.initialize()
	>>> aa_mdl.factorize()
	
	Copyright (C) Christian Thurau, 2010. GNU General Public License (GPL). 
	"""
	
	_VINFO = 'pymf-sub v0.1'
	
	def __init__(self, data, mfmethod, nsub=20, show_progress=True, mapW=False, base_sel=2,
				num_bases=3 , niterH=1, niter=100, compH=True, compW=True, sstrategy='rand'):
		NMF.__init__(self, data, num_bases=num_bases, niter=niter, compH=compH, show_progress=show_progress, compW=compW)
	
		self._niterH = niterH
		self._nsub = nsub
		self.data = data
		self._mfmethod = mfmethod
		self._mapW = mapW
		self._sstrategy = sstrategy
		self._base_sel = base_sel
		
		# assign the correct distance function
		if self._sstrategy == 'cur':
			self._subfunc = self.curselect
					
		elif self._sstrategy == 'kmeans':
			self._subfunc = self.kmeansselect
				
		elif self._sstrategy == 'hull':
			self._subfunc = self.hullselect
			
		else:
			self._subfunc = self.randselect
	
	def hullselect(self):
		
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
				idx = np.append(idx, dist.vq(data[i, :], convex_hull_d.T))
				idx = np.unique(idx)
				
			return np.int32(idx)
	
		
		# determine convex hull data points only if the total
		# amount of available data is >50
		#if self.data.shape[1] > 50:	
		pcamodel = PCA(self.data, show_progress=self._show_progress)		
		pcamodel.factorize()		
	
		idx = selectHullPoints(pcamodel.H, n=self._base_sel)		
		
		# set the number of subsampled data
		self.nsub = len(idx)
		
		return idx

	def kmeansselect(self):
			kmeans_mdl = Kmeans(self.data, num_bases=self._nsub, niter=self._niter)
			kmeans_mdl.initialization()
			kmeans_mdl.factorize()
			
			# pick data samples closest to the centres
			idx = dist.vq(kmeans_mdl.data, kmeans_mdl.W)			
			return idx
			
	def curselect(self):	
		def sample_probability():		
			dsquare = self.data[:,:]**2
						
			pcol = np.array(dsquare.sum(axis=0))				
			pcol /= pcol.sum()	
		
			return (pcol.reshape(-1,1))		
		
		probs = sample_probability()
		prob_cols = np.cumsum(probs.flatten()) #.flatten()				
		temp_ind = np.zeros(self._nsub, np.int32)
	
		for i in range(self._nsub):		 
			tempI = np.where(prob_cols >= np.random.rand())[0]
			temp_ind[i] = tempI[0]    	
			
		return np.sort(temp_ind)
		
	
	def randselect(self):
		idx = random.sample(xrange(self._num_samples), self._nsub)		
		return np.sort(np.int32(idx))
		
	def updateW(self):
		
		idx = self._subfunc()	
		idx = np.sort(np.int32(idx))

		
		mdl_small = self._mfmethod(self.data[:, idx], 
								num_bases=self._num_bases, 
								niter=self._niter, 
								show_progress=self._show_progress, 
								compW=True)

		# initialize W, H, and beta
		mdl_small.initialization()

		# determine W
		mdl_small.factorize()
		
		
		self.mdl = self._mfmethod(self.data[:, :], 
									num_bases=self._num_bases , 
									niter=self._niterH, 
									show_progress=self._show_progress, compW=False)


		self.mdl.initialization()
		
		if self._mapW:
			# compute pairwise distances
			#distance = vq(self.data, self.W)
			_Wmapped_index = dist.vq(self.mdl.data, mdl_small.W)			
			
			# do not directly assign, i.e. Wdist = self.data[:,sel]
			# as self might be unsorted (in non ascending order)
			# -> sorting sel would screw the matching to W if
			# self.data is stored as a hdf5 table (see h5py)
			for i,s in enumerate(_Wmapped_index):
				self.mdl.W[:,i] = self.mdl.data[:,s]
		else:
			self.mdl.W = np.copy(mdl_small.W)
			
	def updateH(self):
		self.mdl.factorize()
		
	def factorize(self):
		"""Do factorization s.t. data = dot(dot(data,beta),H), under the convexity constraint
			beta >=0, sum(beta)=1, H >=0, sum(H)=1
		"""
		# compute new coefficients for reconstructing data points
		self.updateW()		
		
		# for CHNMF it is sometimes useful to only compute
		# the basis vectors
		if self._compH:
			self.updateH()	
			
		self.W = self.mdl.W
		self.H = self.mdl.H
					
		self.ferr = np.zeros(1)
		self.ferr[0] = self.mdl.frobenius_norm()		
		self._print_cur_status(' Fro:' + str(self.ferr[0]))	
		
if __name__ == "__main__":
	import doctest  
	doctest.testmod()	