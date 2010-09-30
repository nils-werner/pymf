#!/bin/python
##    pymf - Python Matrix Factorization library
##    Copyright (C) 2010 Christian Thurau
##
##    This library is free software; you can redistribute it and/or
##    modify it under the terms of the GNU Library General Public
##    License as published by the Free Software Foundation; either
##    version 2 of the License, or (at your option) any later version.
##
##    This library is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
##    Library General Public License for more details.
##
##    You should have received a copy of the GNU Library General Public
##    License along with this library; if not, write to the Free
##    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##
##    Christian Thurau
##    cthurau@gmail.com
"""  

"""

from matplotlib.pyplot import *
import pymf
import time
import numpy as np
import scipy.sparse


def test_svd(A, func, desc, marker):
	stime = time.time()
	m = func(A, show_progress=False, rrank=2, crank=2)	
	m.factorize()
	print desc, m.frobenius_norm()/(A.shape[0] + A.shape[1]) , ' elapsed:' , time.time() - stime
	return m


def test(A, func, desc, marker):
	stime = time.time()
	
	m = func(A, compW=True, num_bases=4, niter=200, show_progress=False)
	m.initialization()	
	m.factorize()
	
	print desc, m.ferr[-1]/(A.shape[0] + A.shape[1]) , ' elapsed:' , time.time() - stime
	return m

def testsub(A, func, mfmethod, nsub, desc, marker):
	stime = time.time()	
	
	m = func(A, mfmethod, compW=True, sstrategy='cur', nsub=nsub, num_bases=4, niter=200, niterH=1,  show_progress=False)
	m.initialization()	
	m.factorize()
	
	print desc, m.ferr[-1]/(A.shape[0] + A.shape[1]) , ' elapsed:' , time.time() - stime
	
	return m

A = np.round(np.random.random((3,100))) + 3.0
B = scipy.sparse.csc_matrix(A)
# test pseudoinverse
#pymf.pinv(A)
#pymf.pinv(B)

svdm = test_svd(A, pymf.SVD, 'svd', 'c<')
svdm = test_svd(B, pymf.SVD, 'svd sparse', 'c<')
curm = test_svd(A, pymf.CUR, 'cur', 'b<')
curm = test_svd(B, pymf.CUR, 'cur sparse', 'b<')
cmdm = test_svd(A, pymf.CMD, 'cmd', 'm<')
cmdm = test_svd(B, pymf.CMD, 'cmd sparse', 'm<')
sparse_svmcur = test_svd(A, pymf.SIVMCUR, 'sivmcur', 'm<')
sparse_svmcur = test_svd(B, pymf.SIVMCUR, 'sivmcur sparse', 'm<')


m = test(A, pymf.PCA, 'pca', 'c<')
m = test(A, pymf.NMF, 'nmf', 'rs')
m = test(A, pymf.SIVM, 'sivm', 'bs')
m = test(A, pymf.Kmeans, 'kmeans', 'b*')
m = test(A, pymf.AA, 'AA', 'bs')#
m = test(A, pymf.SNMF, 'snmf', 'bo')
m = test(A, pymf.CHNMF, 'chnmf', 'm*')
m = test(A, pymf.CNMF, 'cnmf', 'c<')
m = test(A, pymf.BNMF, 'bnmf', 'b>')
m = testsub(A, pymf.SUB, pymf.NMF, 100, 'subnmf', 'c<')