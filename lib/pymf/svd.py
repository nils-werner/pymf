#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: svd.py 24 2010-09-01 07:51:05Z cthurau $
#$Author$
"""  
PyMF Singular Value Decomposition.

	SVD : Class for Singular Value Decomposition
	pinv() : Compute the pseudoinverse of a Matrix
	 
"""

__version__ = "$Revision$"


from numpy.linalg import eigh
import scipy.sparse
import scipy.sparse.linalg.eigen.arpack as arpack
import numpy as np

def pinv(A, eps=10**-8):	
		# calculate SVD
		svd_mdl =  SVD(A)
		svd_mdl.factorize()
		S = svd_mdl.S
		Sdiag = S.diagonal()
		Sdiag = np.where(Sdiag >eps, 1.0/Sdiag, 0.0)
		
		for i in range(S.shape[0]):
			S[i,i] = Sdiag[i]
			
		if scipy.sparse.issparse(A):			
			A_p = svd_mdl.V.T * (S *  svd_mdl.U.T)
		else:
			
			A_p = np.dot(svd_mdl.V.T, np.core.multiply(np.diag(S)[:,np.newaxis], svd_mdl.U.T))

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
		if scipy.sparse.issparse(self.data):
			err = self.data - self.U*self.S*self.V	
			err = err.multiply(err)
			err = np.sqrt(err.sum())
		else:				
			err = self.data[:,:] - np.dot(np.dot(self.U, self.S), self.V)
			err = np.sqrt(np.sum(err**2))
							
		return err
	
	def _eig(self, A, k=1):
		return eigh(A)	
	
	def factorize(self):	
		def _right_svd():			
			AA = np.dot(self.data[:,:], self.data[:,:].T)
			values, u_vectors = self._eig(AA)			
				
			# get rid of too low eigenvalues
			u_vectors = u_vectors[:, values > self._EPS] 
			values = values[values > self._EPS]
							
			# sort eigenvectors according to largest value
			idx = np.argsort(values)
			values = values[idx[::-1]]

			# argsort sorts in ascending order -> access is backwards
			self.U = u_vectors[:,idx[::-1]]
			
			# compute S
			self.S = np.diag(np.sqrt(values))
			
			# and the inverse of it
			S_inv = np.diag(np.sqrt(values)**-1)
					
			# compute V from it
			self.V = np.dot(S_inv, np.dot(self.U[:,:].T, self.data[:,:]))	
			
		
		def _left_svd():
			AA = np.dot(self.data[:,:].T, self.data[:,:])
			values, v_vectors = self._eig(AA)	
		
			# get rid of too low eigenvalues
			v_vectors = v_vectors[:, values > self._EPS] 
			values = values[values > self._EPS]
			
			# sort eigenvectors according to largest value
			# argsort sorts in ascending order -> access is backwards
			idx = np.argsort(values)[::-1]
			values = values[idx]
			
			# compute S
			self.S= np.diag(np.sqrt(values))
			
			# and the inverse of it
			S_inv = np.diag(1.0/np.sqrt(values))	
						
			Vtmp = v_vectors[:,idx]
			
			self.U = np.dot(np.dot(self.data[:,:], Vtmp), S_inv)				
			self.V = Vtmp.T
	
		def _sparse_right_svd():
			## for some reasons arpack does not allow computation of rank(A) eigenvectors (??)						
			values, u_vectors = arpack.eigen_symmetric(self.data*self.data.transpose(), k=self.data.shape[0]-1)							
			
			# get rid of too low eigenvalues
			u_vectors = u_vectors[:, values > self._EPS] 
			values = values[values > self._EPS]
			
			# sort eigenvectors according to largest value
			idx = np.argsort(values)
			values = values[idx[::-1]]						
			
			# argsort sorts in ascending order -> access is backwards			
			self.U = scipy.sparse.csc_matrix(u_vectors[:,idx[::-1]])
					
			# compute S
			self.S = scipy.sparse.csc_matrix(np.diag(np.sqrt(values)))
			
			# and the inverse of it
			S_inv = scipy.sparse.csc_matrix(np.diag(1.0/np.sqrt(values)))			
					
			# compute V from it
			self.V = self.U.transpose() * self.data
			self.V = S_inv * self.V
	
		def _sparse_left_svd():		
			# for some reasons arpack does not allow computation of rank(A) eigenvectors (??)
			values, v_vectors = arpack.eigen_symmetric(self.data.transpose()*self.data,k=self.data.shape[1]-1)
			
			# get rid of too low eigenvalues
			v_vectors = v_vectors[:, values > self._EPS] 
			values = values[values > self._EPS]
			
			# sort eigenvectors according to largest value
			idx = np.argsort(values)
			values = values[idx[::-1]]
			
			# argsort sorts in ascending order -> access is backwards			
			self.V = scipy.sparse.csc_matrix(v_vectors[:,idx[::-1]])
					
			# compute S
			self.S = scipy.sparse.csc_matrix(np.diag(np.sqrt(values)))
			
			# and the inverse of it			
			S_inv = scipy.sparse.csc_matrix(np.diag(1.0/np.sqrt(values)))								
			
			self.U = self.data * self.V * S_inv		
			self.V = self.V.transpose()
	
		
		if self._rows > self._cols:
			if scipy.sparse.issparse(self.data):
				_sparse_left_svd()
			else:			
				_left_svd()
		else:
			if scipy.sparse.issparse(self.data):
				_sparse_right_svd()
			else:			
				_right_svd()
			
if __name__ == "__main__":
	import doctest  
	doctest.testmod()	