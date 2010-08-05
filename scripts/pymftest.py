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
# generate data

def random_gaussians(num=100,dim=100, num_gaussians=4):
	
	def rand_mean(d):
		return np.random.random((d)) * np.random.random_integers(low=0, high=10,size=d)
	
	A = np.random.multivariate_normal(rand_mean(dim), np.diag(np.ones(dim)), num)
	for i in range(num_gaussians-1):	

		B = np.random.multivariate_normal(rand_mean(dim), np.diag(np.ones(dim)), num)
		A = np.concatenate((A,B), axis=0)
	
	return A


s = [2,20]

A = random_gaussians(dim=5,num=20,num_gaussians=4).T + 5


def test_svd(A, func, desc, marker):
	stime = time.time()
	m = func(A, show_progress=False, rrank=2, crank=2)	
	m.factorize()
	print desc, m.frobenius_norm()/(A.shape[0] + A.shape[1]) , ' elapsed:' , time.time() - stime
	plot(m.U[:,0], m.U[:,1], marker, label=desc, ms=10)
	return m


def test(A, func, desc, marker):
	stime = time.time()
	m = func(A, compW=True, num_bases=4, niter=200, show_progress=False)
	m.initialization()	
	m.factorize()
	print desc, m.ferr[-1]/(A.shape[0] + A.shape[1]) , ' elapsed:' , time.time() - stime
	plot(m.W[0,:], m.W[1,:], marker, label=desc, ms=10)
	return m

def testsub(A, func, mfmethod, nsub, desc, marker):
	stime = time.time()	
	m = func(A, mfmethod, compW=True, sstrategy='cur', nsub=nsub, num_bases=4, niter=200, niterH=1,  show_progress=False)
	m.initialization()	
	m.factorize()
	print desc, m.ferr[-1]/(A.shape[0] + A.shape[1]) , ' elapsed:' , time.time() - stime
	plot(m.W[0,:], m.W[1,:], marker, label=desc, ms=10)
	return m

#figure()
#plot(A[0,:], A[1,:], 'g.')
svdm = test_svd(A, pymf.SVD, 'svd', 'c<')
curm = test_svd(A, pymf.CUR, 'cur', 'b<')
cmdm = test_svd(A, pymf.CMD, 'cmd', 'm<')
svmcur = test_svd(A, pymf.SIVMCUR, 'svmcur', 'm<')
m = test(A, pymf.PCA, 'pca', 'c<')
m = test(A, pymf.NMF, 'nmf', 'rs')
m = test(A, pymf.SIVM, 'sivm', 'bs')
m = test(A, pymf.Kmeans, 'kmeans', 'b*')
m = test(A, pymf.AA, 'AA', 'bs')#
m = test(A, pymf.SNMF, 'snmf', 'bo')
m = test(A, pymf.CHNMF, 'chnmf', 'm*')
m = test(A, pymf.CNMF, 'cnmf', 'c<')
m = testsub(A, pymf.SUB, pymf.NMF, 100, 'subnmf', 'c<')
m = test(A, pymf.BNMF, 'bnmf', 'b>')
#legend(loc=2)
#show()
