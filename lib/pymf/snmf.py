#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: snmf.py 20 2010-08-02 17:35:19Z cthurau $
#$Author$
"""  
PyMF Semi Non-negative Matrix Factorization.

	SNMF(NMF) : Class for semi non-negative matrix factorization
	
[1] Ding, C., Li, T. and Jordan, M.. Convex and Semi-Nonnegative Matrix Factorizations.
IEEE Trans. on Pattern Analysis and Machine Intelligence 32(1), 45-55. 
"""

__version__ = "$Revision$"


import numpy as np

from nmf import NMF

__all__ = ["SNMF"]

class SNMF(NMF):
	"""  	
	SNMF(data, num_bases=4, niter=100, show_progress=True, compW=True)
	
	Semi Non-negative Matrix Factorization. Factorize a data matrix into two 
	matrices s.t. F = | data - W*H | is minimal.
	
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
		W : "data_dimension x num_bases" matrix of basis vectors
		H : "num bases x num_samples" matrix of coefficients
	
		ferr : frobenius norm (after calling .factorize())
		
	Example
	-------
	Applying AA to some rather stupid data set:
	
	>>> import numpy as np
	>>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
	>>> snmf_mdl = SNMF(data, num_bases=2, niter=10)
	>>> snmf_mdl.initialization()
	>>> snmf_mdl.factorize()
	
	The basis vectors are now stored in snmf_mdl.W, the coefficients in snmf_mdl.H. 
	To compute coefficients for an existing set of basis vectors simply	copy W 
	to snmf_mdl.W, and set compW to False:
	
	>>> data = np.array([[1.5], [1.2]])
	>>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
	>>> snmf_mdl = SNMF(data, num_bases=2, niter=1, compW=False)
	>>> snmf_mdl.initialization()
	>>> snmf_mdl.W = W
	>>> snmf_mdl.factorize()
	
	The result is a set of coefficients snmf_mdl.H, s.t. data = W * snmf_mdl.H. 
	"""
	
	_VINFO = 'pymf-snmf v0.1'

	def __init__(self, data, num_bases=4, niter=100, show_progress=False, compW=True):
		""" Inits Nmf class:
		
		sampleNmf = Nmf(data, num_bases=4, niter=100, show_progress=True, compW=True)
		
		Args:
			data (required)	: d x n data matrix [d - dimension, n -number of samples]
			num_bases	: number of basis vectors for W (default: 4)
			niter		: number of iterations (default: 100)
			show_progress	: (default: True)
			compW		: set to True if W and H should be optimized, set to False
					if only H should be optimized. This is usefull if W is 
					computed somewhere or if new data should be mapped on a
					given set of basis vectors W.
		"""
		# data can be either supplied by conventional numpy arrays or
		# as a numpy array within a pytables table (should be preferred for large data sets)
		NMF.__init__(self, data, num_bases=num_bases, niter=niter, show_progress=show_progress, compW=compW)
		

	def updateW(self):
		W1 = np.dot(self.data[:,:], self.H.T)
		W2 = np.dot(self.H, self.H.T)	
		W = np.dot(W1, np.linalg.inv(W2))
		
	def updateH(self):
		def separate_positive(m):
			return (np.abs(m) + m)/2.0 
		
		def separate_negative(m):
			return (np.abs(m) - m)/2.0
			
		XW = np.dot(self.data[:,:].T, self.W)				

		WW = np.dot(self.W.T, self.W)
		WW_pos = separate_positive(WW)
		WW_neg = separate_negative(WW)
		
		XW_pos = separate_positive(XW)
		H1 = (XW_pos + np.dot(self.H.T, WW_neg)).T
		
		XW_neg = separate_negative(XW)
		H2 = (XW_neg + np.dot(self.H.T,WW_pos)).T
		
		self.H = np.where(H2 > 0, self.H * np.sqrt(H1/H2), self.H)	
	
if __name__ == "__main__":
	import doctest  
	doctest.testmod()			