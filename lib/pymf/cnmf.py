#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: cnmf.py 20 2010-08-02 17:35:19Z cthurau $
#$Author$
"""  	
PyMF Convex Matrix Factorization [1]

	CNMF(NMF) : Class for convex matrix factorization
	
[1] Ding, C., Li, T. and Jordan, M.. Convex and Semi-Nonnegative Matrix Factorizations.
IEEE Trans. on Pattern Analysis and Machine Intelligence 32(1), 45-55.
"""

__version__ = "$Revision$"

import numpy as np

from nmf import NMF
from kmeans import Kmeans


__all__ = ["CNMF"]

class CNMF(NMF):
	"""  	
	CNMF(data, num_bases=4, niter=100, show_progress=True, compW=True)
	
	
	Convex NMF. Factorize a data matrix into two matrices s.t.
	F = | data - W*H | = | data - data*beta*H| is minimal. H and beta
	are restricted to convexity (beta >=0, sum(beta, axis=1) = [1 .. 1]).	
	
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
	Applying CNMF to some rather stupid data set:
	
	>>> import numpy as np
	>>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
	>>> cnmf_mdl = CNMF(data, num_bases=2, niter=10)
	>>> cnmf_mdl.initialization()
	>>> cnmf_mdl.factorize()
	
	The basis vectors are now stored in cnmf_mdl.W, the coefficients in cnmf_mdl.H. 
	To compute coefficients for an existing set of basis vectors simply	copy W 
	to cnmf_mdl.W, and set compW to False:
	
	>>> data = np.array([[1.5, 1.3], [1.2, 0.3]])
	>>> W = [[1.0, 0.0], [0.0, 1.0]]
	>>> cnmf_mdl = CNMF(data, num_bases=2, niter=1, compW=False)
	>>> cnmf_mdl.initialization()
	>>> cnmf_mdl.W = W
	>>> cnmf_mdl.factorize()
	
	The result is a set of coefficients acnmf_mdl.H, s.t. data = W * cnmf_mdl.H.
	"""

	_VINFO = 'pymf-cnmf v0.1'
	
	def __init__(self, data, num_bases=4, niter=100, show_progress=False, compW=True):
		# data can be either supplied by conventional numpy arrays or
		# as a numpy array within a pytables table (should be preferred for large data sets)
		NMF.__init__(self, data, num_bases=num_bases, niter=niter, show_progress=show_progress, compW=compW)
	
	
	def initialization(self):		
		# init basic matrices		
		self.W = np.zeros((self._data_dimension, self._num_bases))
		self.H = np.zeros((self._num_bases, self._num_samples))
		self.G = np.zeros((self._num_samples, self._num_bases))
		#####
		
		# initialize using k-means
		km = Kmeans(self.data[:,:], num_bases=self._num_bases, show_progress=self._show_progress)
		km.initialization()
		km.factorize()
		assign = km.assigned

		num_i = np.zeros(self._num_bases)
		for i in range(self._num_bases):
			num_i[i] = len(np.where(assign == i)[0])

		self.G[range(len(assign)), assign] = 1.0 
		self.G += 0.01						
		self.G /= np.tile(np.reshape(num_i[assign],(-1,1)), self.G.shape[1])
	
		self.H.T[range(len(assign)), assign] = 1.0				
		self.H += 0.2*np.ones((self._num_bases, self._num_samples))
		
		self.W = np.dot(self.data[:,:], self.G)
			
	
	# see .factorize() for the update of W and H
	def updateW(self):
		pass		
		
	def updateH(self):
		pass
	
	def factorize(self):
		""" Perform factorization s.t. data = WH using the standard multiplicative 
		update rule for Non-negative matrix factorization.
		"""	
		def separate_positive(m):
			return (np.abs(m) + m)/2.0 
		
		def separate_negative(m):
			return (np.abs(m) - m)/2.0 
			
		XtX = np.dot(self.data[:,:].T, self.data[:,:])
		XtX_pos = separate_positive(XtX)
		XtX_neg = separate_negative(XtX)
							
		# iterate over W and H
		for i in xrange(self._niter):
			# update H
			H_x_WT = np.dot(self.H.T, self.G.T)
				
			XtX_neg_x_W = np.dot(XtX_neg, self.G)
			XtX_pos_x_W = np.dot(XtX_pos, self.G)
		
			ha = XtX_pos_x_W + np.dot(H_x_WT, XtX_neg_x_W)
			hb = XtX_neg_x_W + np.dot(H_x_WT, XtX_pos_x_W) + 10**-9
			self.H = (self.H.T*np.sqrt(ha/hb)).T
		
			# update W			
			if self._compW:
				HT_x_H = np.dot(self.H, self.H.T)
				wa = np.dot(XtX_pos, self.H.T) + np.dot(XtX_neg_x_W, HT_x_H)
				wb = np.dot(XtX_neg, self.H.T) + np.dot(XtX_pos_x_W, HT_x_H) + 10**-9
			
				self.G *= np.sqrt(wa/wb)
				self.W = np.dot(self.data[:,:], self.G)
								
			self.ferr[i] = self.frobenius_norm()

			self._print_cur_status('iteration ' + str(i+1) + '/' + str(self._niter) + ' Fro:' + str(self.ferr[i]))

			if i > 1:
				if self.converged(i):					
					self.ferr = self.ferr[:i]					
					break

if __name__ == "__main__":
	import doctest  
	doctest.testmod()	
