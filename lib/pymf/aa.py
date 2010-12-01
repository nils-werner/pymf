#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: aa.py 21 2010-08-05 08:13:08Z cthurau $
#$Author$
"""  
PyMF Archetypal Analysis [1]

	AA: class for Archetypal Analysis

[1] Cutler, A. Breiman, L. (1994), "Archetypal Analysis", Technometrics 36(4), 338-347.
"""

__version__ = "$Revision$"

import numpy as np
import random
from dist import vq
from cvxopt import solvers, base

#from numpy.linalg import pinv
from svd import pinv
from nmf import NMF

__all__ = ["AA"]

class AA(NMF):	
	"""  	
	AA(data, num_bases=4, niter=100, show_progress=True, compW=True)
	
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
		Compute W (True) or only H (False). Useful for using precomputed
		basis vectors.
		
	Attributes
	----------
		W : "data_dimension x num_bases" matrix of basis vectors
		H : "num bases x num_samples" matrix of coefficients
		beta : "num_bases x num_samples" matrix of basis vector coefficients 
			(for constructiong W s.t. W = beta * data.T )
			
		ferr : frobenius norm (after calling .factorize())
	
	Example
	-------
	Applying AA to some rather stupid data set:
	
	>>> import numpy as np
	>>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
	>>> aa_mdl = AA(data, num_bases=2, niter=10)
	>>> aa_mdl.initialization()
	>>> aa_mdl.factorize()
	
	The basis vectors are now stored in aa_mdl.W, the coefficients in aa_mdl.H. 
	To compute coefficients for an existing set of basis vectors simply	copy W 
	to aa_mdl.W, and set compW to False:
	
	>>> data = np.array([[1.5], [1.2]])
	>>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
	>>> aa_mdl = AA(data, num_bases=2, niter=1, compW=False)
	>>> aa_mdl.initialization()
	>>> aa_mdl.W = W
	>>> aa_mdl.factorize()
	
	The result is a set of coefficients aa_mdl.H, s.t. data = W * aa_mdl.H.
	"""
	# set cvxopt options
	solvers.options['show_progress'] = False
		
		
	def __init__(self, data, num_bases=4, niter=100, show_progress=False, compW=True):

		# call inherited method
		NMF.__init__(self, data, num_bases=num_bases, niter=niter, show_progress=show_progress, compW=compW)


	def initialization(self): 
		""" Initialize W and H to random values in [0,1]
		"""
		# initialize beta coefficients -> convex factors for creating basis vectors from data samples
		self.H = np.random.random((self._num_bases, self._num_samples))

		self.H /= self.H.sum(axis=0)
			
		# reintialize W to random data values
		self.beta = np.random.random((self._num_bases, self._num_samples))
		self.beta /= self.beta.sum(axis=0)
		self.W = np.dot(self.beta, self.data.T).T


	def map_W_to_Data(self):
		""" Return data points that are most similar to basis vectors W
		"""
		
		# compute pairwise distances
		#distance = vq(self.data, self.W)
		self._Wmapped_index = vq(self.data, self.W)
		self.Wmapped = np.zeros(self.W.shape)
		
		# do not directly assign, i.e. Wdist = self.data[:,sel]
		# as self might be unsorted (in non ascending order)
		# -> sorting sel would screw the matching to W if
		# self.data is stored as a hdf5 table (see h5py)
		for i,s in enumerate(self._Wmapped_index):
			self.Wmapped[:,i] = self.data[:,s]


	def updateH(self):
			def updatesingleH(i):
				# optimize alpha using qp solver from cvxopt
				FA = base.matrix(np.float64(np.dot(-self.W.T, self.data[:,i])))
				al = solvers.qp(HA, FA, INQa, INQb, EQa, EQb)
				self.H[:,i] = np.array(al['x']).reshape((1,self._num_bases))
			
			EQb = base.matrix(1.0, (1,1))
			# float64 required for cvxopt
			HA = base.matrix(np.float64(np.dot(self.W.T, self.W)))			
			INQa = base.matrix(-np.eye(self._num_bases))
			INQb = base.matrix(0.0, (self._num_bases,1))
			EQa = base.matrix(1.0, (1, self._num_bases))
	
			map(updatesingleH, xrange(self._num_samples))


	def updateW(self):
			def updatesingleW(i):
				# optimize beta	 using qp solver from cvxopt
				FB = base.matrix(np.float64(np.dot(-self.data.T, W_hat[:,i])))				
				be = solvers.qp(HB, FB, INQa, INQb, EQa, EQb)
				self.beta[i,:] = np.array(be['x']).reshape((1,self._num_samples))
				
			# float64 required for cvxopt
			HB = base.matrix(np.float64(np.dot(self.data[:,:].T, self.data[:,:])))					
			EQb = base.matrix(1.0, (1,1))
			W_hat = np.dot(self.data, pinv(self.H))			
			INQa = base.matrix(-np.eye(self._num_samples))
			INQb = base.matrix(0.0, (self._num_samples,1))
			EQa = base.matrix(1.0, (1,self._num_samples))
		
			map(updatesingleW, xrange(self._num_bases))	
											
			self.W = np.dot(self.beta,self.data.T).T		
			
			
if __name__ == "__main__":
	import doctest  
	doctest.testmod()			
