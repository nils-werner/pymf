#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: cmd.py 24 2010-09-01 07:51:05Z cthurau $
#$Author$
"""
PyMF Compact Matrix Decomposition [1]

	CMD(CUR):  Class for Compact Matrix Decomposition

[1] Sun, J., Xie, Y., Zhang, H. and Faloutsos, C. (2007), Less is More: Compact Matrix Decomposition for Large
Sparse Graphs, in Proc. SIAM Int. Conf. on Data Mining. 
"""

__version__ = "$Revision$"

import numpy as np
from cur import CUR

__all__ = ["CMD"]

class CMD(CUR):
	"""  	
	CMD(data, rrank=0, crank=0, show_progress=False)
	
	
	Compact Matrix Decomposition. Factorize a data matrix into three matrices s.t.
	F = | data - USV| is minimal. CMD randomly selects rows and columns from
	data for building U and V, respectively. 
	
	Parameters
	----------
	data : array_like
		the input data
	rrank: int, optional 
		Number of rows to sample from data. Double entries are eliminiated s.t.
		the resulting rank might be lower.
		4 (default)
	crank: int, optional
		Number of columns to sample from data. Double entries are eliminiated s.t.
		the resulting rank might be lower.
		4 (default)
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
	>>> cmd_mdl = CMD(data, show_progress=False, rrank=1, crank=2)	
	>>> cmd_mdl.factorize()
	"""
	
	_VINFO = 'pymf-cmd v0.1'
	

	def __init__(self, data, rrank=0, crank=0, show_progress=True):
		CUR.__init__(self, data, rrank=rrank, crank=rrank, show_progress=show_progress)
	
	def _cmdinit(self):
		nrids = np.unique(self._rid)
		ncids = np.unique(self._cid)
	
		self._rcnt = np.zeros(len(nrids))		
		self._ccnt = np.zeros(len(ncids))
			
		for i,idx in enumerate(nrids):
			self._rcnt[i] = len(np.where(self._rid == idx)[0])
	
		for i,idx in enumerate(ncids):
			self._ccnt[i] = len(np.where(self._cid == idx)[0])

		self._rid = np.int32(list(nrids))
		self._cid = np.int32(list(ncids))
	
	def factorize(self):								
		[prow, pcol] = self.sample_probability()
		self._rid = self.sample(self._rrank, prow)
		self._cid = self.sample(self._crank, pcol)
							
		self._cmdinit()
		self.computeUCR()
		
		
if __name__ == "__main__":
	import doctest  
	doctest.testmod()			