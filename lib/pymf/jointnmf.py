#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: nmf.py 20 2010-08-02 17:35:19Z cthurau $
#$Author$
"""  	
PyMF Joint Non-negative Matrix Factorization.

	JointNMF: Class for Joint Non-negative Matrix Factorization

[1] Lee, D. D. and Seung, H. S. (1999), Learning the Parts of Objects by Non-negative 
Matrix Factorization, Nature 401(6755), 788-799.
"""

__version__ = "$Revision$"


import numpy as np

from nmf import NMF
__all__ = ["JointNMF"]

class JointNMF(NMF):
	"""  	
	JointNMF(data_1, data_2, lambd=0.5, num_bases=4, 
		niter=100,	show_progress=False, compH=True, compW=True)
	
	
	Joint Non-negative Matrix Factorization. Factorize two data matrices into three matrices 
	s.t. F = L*| data_1 - W_1*H | + (1-L)| data_2 - W_2*H| is minimal. H, and W are 
	restricted to non-negative data (uses the classicial multiplicative update rule).
	"data_1" and "data_2" must have the same number of columns.
	
	Parameters
	----------
	data_1 : array_like 
		the input data 1
	data_2 : array_like
		the input data 2
	lamb_d: float [0.0, 1.0]
		mixing coefficient, controls contribution of data_1 vs. data_2
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
		W: concatenated basis vectors W_1 and W_2
		   W_1 "data_dimension_1 x num_bases" matrix of basis vectors
		   W_2"data_dimension_2 x num_bases" matrix of basis vectors
		H : "num bases x num_samples" matrix of coefficients	
	
	Example
	-------
	Applying JointNMF to some rather stupid data set:
	
	>>> import numpy as np
	>>> data_1 = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
	>>> data_1 = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
	>>> nmf_mdl = JointNMF(data_1, data_2, lambd=0.5, num_bases=2, niter=10)
	>>> nmf_mdl.initialization()
	>>> nmf_mdl.factorize()
	
	
	The basis vectors are now stored in nmf_mdl.W, the coefficients in nmf_mdl.H. 
	To compute coefficients for an existing set of basis vectors simply	copy W 
	to nmf_mdl.W, and set compW to False [-> use NMF not JointNMF !!!]
	
	>>> data_1 = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
	>>> W = np.array([[1.0, 0.0], [0.0, 1.0]],[[1.0, 0.3], [0.3, 1.0], [0.1, 0.6]])
	>>> nmf_mdl = NMF(data_1, num_bases=2, niter=100, compW=False)
	>>> nmf_mdl.initialization()
	>>> nmf_mdl.W = W
	>>> nmf_mdl.factorize()
	
	Note that for the dimensionality of nmf_mdl.W we have: 
	nmf_mdl.W.shape[0] = data_1_dim + data_2_dim
	Thus, the resulting coefficients (nmf_mdl.H) can be used to reconstruct the 
	unobserved data_2, => data_2 = np.dot(nmf_mdl.W[data_1_dim:,:], nmf_mdl.H) 		
	"""
	
	_VINFO = 'pymf-jointnmf v0.1'
	
	EPS = 10**-8
	
	def __init__(self, data_1, data_2, lambd=0.5, num_bases=4, niter=100, show_progress=False, compH=True, compW=True):
		
		# generate a new data set data using a weighted
		# combination of data_1 and data_2
		
		self._data_1 = data_1
		self._data_2 = data_2
		self._lambd = lambd
		
		data = np.concatenate((lambd*self._data_1, (1.0-lambd)* self._data_2), axis=0)		
		NMF.__init__(self, data, num_bases=num_bases, niter=niter, show_progress=show_progress, compW=compW)

			
	def factorize(self):
		""" Perform factorization s.t. data = WH using the standard multiplicative 
		update rule for Non-negative matrix factorization.
		"""	
							
		# iterate over W and H
		# build weighted matrix data
		self.data = np.concatenate((self._lambd*self._data_1, (1.0-self._lambd)* self._data_2), axis=0)
		
		for i in xrange(self._niter):
			# update H
			self.updateH()
		
			# update W
			if self._compW:
				self.updateW()
								
			self.ferr[i] = self.frobenius_norm()		
											
			self._logger.info('iteration ' + str(i+1) + '/' + str(self._niter) + ' Fro:' + str(self.ferr[i]))
			
					# check if the err is not changing anymore
			if i > 1:
				if self.converged(i):		
					# adjust the error measure
					self.ferr = self.ferr[:i]			
					break
				
		# rescale the data and the basis vector matrices (coefficients still match)
		self.data = np.concatenate((self._data_1, self._data_2), axis=0)
		self.W[:self._data_1.shape[0],:]/= self._lambd
		self.W[self._data_1.shape[0]:,:]/= (1.0 - self._lambd)		

if __name__ == "__main__":
	import doctest  
	doctest.testmod()	
