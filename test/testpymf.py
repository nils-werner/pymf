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

import pymf
import time
import numpy as np
import scipy.sparse


def test_svd(A, func, desc, marker):
	stime = time.time()
	m = func(A, show_progress=False, rrank=2, crank=2)	
	m.factorize()
	print desc + ': Fro.:', m.frobenius_norm()/(A.shape[0] + A.shape[1]) , ' elapsed:' , time.time() - stime
	return m


def test(A, func, desc, marker, niter=200):
	stime = time.time()
	
	m = func(A, compW=True, num_bases=4, niter=niter, show_progress=False)
	m.initialization()	
	m.factorize()
	
	print desc + ': Fro.:', m.ferr[-1]/(A.shape[0] + A.shape[1]) , ' elapsed:' , time.time() - stime
	return m

def testsub(A, func, mfmethod, nsub, desc, marker):
	stime = time.time()	
	
	m = func(A, mfmethod, compW=True, sstrategy='cur', nsub=nsub, num_bases=8, niter=200, niterH=1,  show_progress=False)
	m.initialization()	
	m.factorize()
	
	print desc, m.ferr[-1]/(A.shape[0] + A.shape[1]) , ' elapsed:' , time.time() - stime
	
	return m

print "test all methods on boring random data..."
A = np.round(np.random.random((2,200))) + 2.0
B = scipy.sparse.csc_matrix(A)
# test pseudoinverse
pymf.pinv(A)
pymf.pinv(B)
svdm = test_svd(A, pymf.SVD, 'Singula Value Decomposition (SVD)', 'c<')
#svdm = test_svd(B, pymf.SVD, 'svd sparse', 'c<')
curm = test_svd(A, pymf.CUR, 'CUR Matrix Decomposition', 'b<')
curm = test_svd(B, pymf.CUR, 'CUR Matrix Decomposition (sparse data)', 'b<')
cmdm = test_svd(A, pymf.CMD, 'Compact Matrix Decomposition (CMD)', 'm<')
cmdm = test_svd(B, pymf.CMD, 'Compact Matrix Decomposition (CMD - sparse data)', 'm<')
sparse_svmcur = test_svd(A, pymf.SIVMCUR, 'Simplex Volume Maximization f. CUR (SIVMCUR)', 'm<')
sparse_svmcur = test_svd(B, pymf.SIVMCUR, 'Simplex Volume Maximization f. CUR (SIVMCUR - sparse data)', 'm<')
m = test(A, pymf.PCA, 'Principal Component Analysis (PCA)', 'c<')
m = test(A, pymf.NMF, 'Non-negative Matrix Factorization (NMF)', 'rs')
m = test(A, pymf.NMFALS, 'NMF u. alternating least squares (NMFALS)', 'rs', niter=10)
m = test(A, pymf.NMFNNLS, 'NMF u. non-neg. least squares (NMFNNLS)', 'rs', niter=10)
m = test(A, pymf.LAESA, 'Linear Approximating Eliminating Search Algorithm (LAESA)', 'rs')
m = test(A, pymf.SIVM, 'Simplex Volume Maximization (SIVM)', 'bs')
m = test(A, pymf.Kmeans, 'K-means clustering (Kmeans)', 'b*')
m = test(A, pymf.Cmeans, 'C-means clustering (Cmeans)', 'b*')
m = test(A, pymf.AA, 'Archetypal Analysis (AA)', 'bs')
m = test(A, pymf.SNMF, 'Semi Non-negative Matrix Factorization (SNMF)', 'bo')
m = test(A, pymf.CNMF, 'Convex non-negative Matrix Factorization (CNMF)', 'c<')
m = test(A, pymf.CHNMF, 'Convex-hull non-negative Matrix Factorization (CHNMF)', 'm*')
m = test(np.round(A-2.0), pymf.BNMF, 'Binary Matrix Factorization (BNMF)', 'b>')