#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: svd.py 20 2010-08-02 17:35:19Z cthurau $
#$Author$
"""  
PyMF Singular Value Decomposition.

	SVD : Class for Singular Value Decomposition
	pinv() : Compute the pseudoinverse of a Matrix
	 
"""

__version__ = "$Revision$"


import numpy as np
from numpy.linalg import eigh

__all__ = ["SVD", "pinv"]

def pinv(A, eps=10**-8):	
		# calculate SVD
		svd_mdl =  SVD(A)
		svd_mdl.factorize()
		
		# calculate pseudoinverse
		S = np.diag(svd_mdl.S).transpose()
		S = np.where(S>eps, 1.0/S, 0.0)
			
		A_p = np.zeros((svd_mdl.V.shape[1], svd_mdl.U.shape[0]))
		A_p[:,:] = np.dot(np.dot(svd_mdl.V.T,  np.diag(S)),svd_mdl.U.T)
		
		return A_p	

class SVD():	
	"""  	
	SVD(data, show_progress=False)
	
	
	Singular Value Decomposition. Factorize a data matrix into three matrices s.t.
	F = | data - USV| is minimal. U and V correspond to eigenvectors of the matrices
	data*data.T and data.T*data.
	
	Parameters
	----------
	data : array_like
		the input data
	show_progress: bool, optional
		Print some extra information
		False (default)	
	
	Attributes
	----------
		U,S,V : submatrices s.t. data = USV				
	
	Example
	-------
	>>> import numpy as np
	>>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
	>>> svd_mdl = SVD(data, show_progress=False)	
	>>> svd_mdl.factorize()
	"""
	
	_EPS=10**-8
	
	_VINFO = 'pymf-svd v0.1'
	
	def __init__(self, data, rrank=0, crank=0, show_progress=True):
		self.data = data
		(self._rows, self._cols) = self.data.shape
		if rrank > 0:
			self._rrank = rrank
		else:
			self._rrank = self._rows
			
		if crank > 0:			
			self._crank = crank
		else:
			self._crank = self._cols

		self._show_progress = show_progress
	
	def frobenius_norm(self):
		""" Frobenius norm (||data - USV||) for a data matrix and a low rank
		approximation given by SVH using rank k for U and V
		
		Returns:
			frobenius norm: F = ||data - USV||
		"""
		# set the rank for the SVD approximation
		err = np.sqrt(np.sum((self.data[:,:] - np.dot(np.dot(self.U, self.S), self.V))**2))
		return err
	
	def rel_norm(self, rset=[-1], cset=[-1]):
			rerr = self.frobenius_norm()**2 / np.sum(self.data[:,:]**2)
			return rerr
	
	def factorize(self):	
		def _right_svd():			
			AA = np.dot(self.data[:,:], self.data[:,:].T)
			values, u_vectors = eigh(AA)			
							
			# sort eigenvectors according to largest value
			idx = np.argsort(values)
			values = values[idx[::-1]]

			# only select eigenvalues >eps
			sel = np.where(values>self._EPS)[0]		
			values = values[sel]	
						
			# argsort sorts in ascending order -> access is backwards
			self.U = u_vectors[:,idx[::-1]][:,sel]
			
			# compute S
			self.S = np.diag(np.sqrt(values))
			
			# and the inverse of it
			S_inv =np.diag(np.sqrt(values)**-1)
					
			# compute V from it
			self.V = np.dot(S_inv, np.dot(self.U[:,:].T, self.data[:,:]))	
			
		
		def _left_svd():
			AA = np.dot(self.data[:,:].T, self.data[:,:])
			values, v_vectors = eigh(AA)	
			
			# sort eigenvectors according to largest value
			# argsort sorts in ascending order -> access is backwards
			idx = np.argsort(values)[::-1]
			values = values[idx]
		
			# only select eigenvalues >eps
			sel = np.where(values>self._EPS)
			idx = idx[sel]
			values = values[sel]
			
			# compute S
			self.S= np.diag(np.sqrt(values))
			
			# and the inverse of it
			S_inv = np.diag(1.0/np.sqrt(values))	
						
			Vtmp = v_vectors[:,idx]
			
			self.U = np.dot(np.dot(self.data[:,:], Vtmp), S_inv)				
			self.V = Vtmp.T
	
		
		if self._rows > self._cols:			
			_left_svd()
		else:
			_right_svd()
			
if __name__ == "__main__":
	import doctest  
	doctest.testmod()	