#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: bnmf.py 20 2010-08-02 17:35:19Z cthurau $
#$Author$
"""
PyMF Binary Matrix Factorization [1]

	BNMF(NMF) : Class for binary matrix factorization

[1]Z. Zhang, T. Li, C. H. Q. Ding, X. Zhang: Binary Matrix Factorization with 
Applications. ICDM 2007
"""

__version__ = "$Revision$"


import numpy as np
from nmf import NMF

__all__ = ["BNMF"]

class BNMF(NMF):
	"""  	
	BNMF(data, num_bases=4, niter=100, show_progress=True, compW=True)
		
	Binary Matrix Factorization. Factorize a data matrix into two matrices s.t.
	F = | data - W*H | is minimal. H and W are restricted to binary values.
	
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
		from another matrix factorization function.
	
	Attributes
	----------
		W : "data_dimension x num_bases" matrix of basis vectors
		H : "num bases x num_samples" matrix of coefficients	
		ferr : frobenius norm (after calling .factorize())
	
	Example
	-------
	Applying BNMF to some rather stupid data set:
	
	>>> import numpy as np
	>>> data = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
	>>> bnmf_mdl = BNMF(data, num_bases=2, niter=10)
	>>> bnmf_mdl.initialization()
	>>> bnmf_mdl.factorize()
	
	The basis vectors are now stored in bnmf_mdl.W, the coefficients in bnmf_mdl.H. 
	To compute coefficients for an existing set of basis vectors simply	copy W 
	to bnmf_mdl.W, and set compW to False:
	
	>>> data = np.array([[0.0], [1.0]])
	>>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
	>>> bnmf_mdl = BNMF(data, num_bases=2, niter=1, compW=False)
	>>> bnmf_mdl.initialization()
	>>> bnmf_mdl.W = W
	>>> bnmf_mdl.factorize()
	
	The result is a set of coefficients bnmf_mdl.H, s.t. data = W * bnmf_mdl.H.
	"""
	
	def __init__(self, data, num_bases=4, niter=100, show_progress=False, compW=True):		
		# data can be either supplied by conventional numpy arrays or
		# as a numpy array within a pytables table (should be preferred for large data sets)
		
		NMF.__init__(self, data, num_bases=num_bases, niter=niter, show_progress=show_progress, compW=compW)
		
		# controls how fast lambda should increase:
		# this influence convergence to binary values during the update. A value
		# <1 will result in non-binary decompositions as the update rule effectively
		# is a conventional nmf update rule. Values >1 give more weight to making the
		# factorization binary with increasing iterations.
		# setting either W or H to 0 results make the resulting matrix non binary.
		self._lamb_increase_W = 1.1 
		self._lamb_increase_H = 1.1 

	def updateH(self):
			H1 = np.dot(self.W.T, self.data[:,:]) + 3.0*self._lamb_H*(self.H**2)
			H2 = np.dot(np.dot(self.W.T,self.W), self.H) + 2*self._lamb_H*(self.H**3) + self._lamb_H*self.H + 10**-9
			self.H *= H1/H2

	def updateW(self):
			W1 = np.dot(self.data[:,:], self.H.T) + 3.0*self._lamb_W*(self.W**2)
			W2 = np.dot(self.W, np.dot(self.H, self.H.T)) + 2.0*self._lamb_W*(self.W**3) + self._lamb_W*self.W  + 10**-9
			self.W *= W1/W2

	def factorize(self):
		""" Perform factorization s.t. data = WH using the standard multiplicative 
		update rule for binary matrix factorization.
		"""		
		
		self._lamb_W = 1.0/self._niter
		self._lamb_H = 1.0/self._niter		
		

		for i in range(self._niter):
			self.ferr[i] = self.frobenius_norm()
			
			if self.updateW:
				self.updateW()
				
			self.updateH()

			epsilon = np.sum((self.H**2 - self.H)**2) + np.sum((self.W**2 - self.W)**2)
			
			if epsilon < 10**-10:
				break
			else:
				self._lamb_W = self._lamb_increase_W * self._lamb_W
				self._lamb_H = self._lamb_increase_H * self._lamb_H
			
			self._logger.info('iteration ' + str(i+1) + '/' + str(self._niter) + ' FN:' + str(self.ferr[i]))


if __name__ == "__main__":
	import doctest  
	doctest.testmod()	
