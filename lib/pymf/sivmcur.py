#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: sivmcur.py 24 2010-09-01 07:51:05Z cthurau $
#$Author$
"""  
PyMF Simplex Volume Maximization for CUR [1]

	SIVMCUR: class for SiVMCUR

[1] C. Thurau, K. Kersting, and C. Bauckhage. Yes We Can - Simplex Volume 
Maximization for Descriptive Web-Scale Matrix Factorization. In Proc. Int. 
Conf. on Information and Knowledge Management. ACM. 2010.
"""

__version__ = "$Revision$"


import numpy as np
from sivm import SIVM
from cur import CUR

__all__ = ["SIVMCUR"]

class SIVMCUR(CUR):
	'''
	SIVMCUR(data, num_bases=4, show_progress=True, compW=True, product=False, dist_measure='l2')
	
	Simplex Volume based CUR Decomposition. Factorize a data matrix into three 
	matrices s.t. F = | data - USV| is minimal. Unlike CUR, SIVMCUR selects the
	rows and columns using SIVM, i.e. it tries to maximize the volume of the
	enclosed simplex.
	
	Parameters
	----------
	data : array_like
		the input data
	rrank: int, optional 
		Number of rows to sample from data.
		4 (default)
	crank: int, optional
		Number of columns to sample from data.
		4 (default)
	product: bool, optional
		Use the alternative product-rule for maximizing the simplex. 
		False (default)
	show_progress: bool, optional
		Print some extra information
		False (default)	
	dist_measure: string, optional
		The distance measure for finding the next best candidate that 
		maximizes the simplex volume ['l2','l1','cosine','sparse_graph_l2']
		'l2' (default)
	
	Attributes
	----------
		U,S,V : submatrices s.t. data = USV		
	
	Example
	-------
	>>> import numpy as np
	>>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
	>>> sivmcur_mdl = SIVMCUR(data, show_progress=False, rrank=1, crank=2)	
	>>> sivmcur_mdl.factorize()
	'''
	
	_VINFO = 'pymf-sivmcur v0.1'
	
	def __init__(self, data, rrank=0, crank=0, show_progress=True, product=False, dist_measure='l2'):
		CUR.__init__(self, data, rrank=rrank, crank=rrank, show_progress=show_progress)
	
		self._product = product
		self._dist_measure = dist_measure	

		
	def sample(self, A, c):
		# switch between product rule and sum rule	
		# !!! NOT YET IMPLEMENTED !!!	
		if self._product:
			m = SIVM
		else:
			m = SIVM
			
		sivmnmf_mdl = m(A, num_bases=c, compH=False, show_progress=self._show_progress, dist_measure=self._dist_measure)
			
		sivmnmf_mdl.initialization()
		sivmnmf_mdl.factorize()
		
		return sivmnmf_mdl.select	
			
			
	def factorize(self):			
		self._rid = self.sample(self.data.transpose(), self._rrank)
		self._cid = self.sample(self.data, self._crank)
						
		self._rcnt = np.ones(len(self._rid))
		self._ccnt = np.ones(len(self._cid))	
									
		self.computeUCR()


if __name__ == "__main__":
	import doctest  
	doctest.testmod()					