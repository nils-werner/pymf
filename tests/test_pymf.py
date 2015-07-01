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

import pytest
import pymf
import time
import numpy as np
import scipy.sparse


np.random.seed(400401)
A = np.random.random((3, 50)) + 2.0
B = scipy.sparse.csc_matrix(A)


@pytest.mark.parametrize("A", [A, B])
def test_pinv(A):
    pymf.pinv(A)


@pytest.mark.parametrize("A,func", [
    (A, pymf.SVD),  # 'Singula Value Decomposition (SVD)', 'c<'
    (A.T, pymf.SVD),  # 'Singula Value Decomposition (SVD)', 'c<'
    (B, pymf.SVD),  # 'svd sparse', 'c<'
    (A, pymf.CUR),  # 'CUR Matrix Decomposition', 'b<'
    (B, pymf.CUR),  # 'CUR Matrix Decomposition (sparse data)', 'b<'
    (A, pymf.CMD),  # 'Compact Matrix Decomposition (CMD)', 'm<'
    (B, pymf.CMD),  # 'Compact Matrix Decomposition (CMD - sparse data)', 'm<'
    (A, pymf.SIVM_CUR),  # 'Simplex Volume Maximization f. CUR (SIVMCUR)', 'm<'
    (A, pymf.SIVM_CUR),  # 'Simplex Volume Maximization f. CUR (SIVMCUR)', 'm<'
])
def test_svd(A, func):
    stime = time.time()
    m = func(A, rrank=2, crank=2)
    m.factorize()
    fro_norm = m.frobenius_norm()/(A.shape[0] + A.shape[1])

    assert fro_norm < 0.1
    print 'Fro.: %d, elapsed %d' % (fro_norm, time.time() - stime)


@pytest.mark.parametrize("A,func,niter,num_bases", [
    (A, pymf.SIVM_SEARCH, 20, 2),  # 'SIVM_SEARCH', 'c<', num_bases=2
    (A, pymf.SIVM_GSAT, 20, 4),  # 'SIVM_GSAT ', 'c<'
    (A, pymf.SIVM_SGREEDY, 20, 4),  # 'SIVM Greedy ', 'c<'
    (A, pymf.GMAP, 20, 4),  # 'GMAP ', 'c<'
    (A, pymf.PCA, 20, 4),  # 'Principal Component Analysis (PCA)', 'c<'
    (A, pymf.NMF, 20, 4),  # 'Non-negative Matrix Factorization (NMF)', 'rs'
    (A, pymf.NMFALS, 10, 4),  # 'NMF u. alternating least squares (NMFALS)', 'rs', niter=10
    (A, pymf.NMFNNLS, 10, 4),  # 'NMF u. non-neg. least squares (NMFNNLS)', 'rs', niter=10
    (A, pymf.LAESA, 20, 4),  # 'Linear Approximating Eliminating Search Algorithm (LAESA)', 'rs'
    (A, pymf.SIVM, 20, 4),  # 'Simplex Volume Maximization (SIVM)', 'bs'
    (A, pymf.Kmeans, 20, 4),  # 'K-means clustering (Kmeans)', 'b*'
    (A, pymf.Cmeans, 20, 4),  # 'C-means clustering (Cmeans)', 'b*'
    (A, pymf.AA, 20, 4),  # 'Archetypal Analysis (AA)', 'bs'
    (A, pymf.SNMF, 20, 4),  # 'Semi Non-negative Matrix Factorization (SNMF)', 'bo'
    (A, pymf.CNMF, 20, 4),  # 'Convex non-negative Matrix Factorization (CNMF)', 'c<'
    (A, pymf.CHNMF, 20, 4),  # 'Convex-hull non-negative Matrix Factorization (CHNMF)', 'm*'
    (np.round(A-2.0), pymf.BNMF, 20, 4),  # 'Binary Matrix Factorization (BNMF)', 'b>'
])
def test(A, func, niter, num_bases):
    stime = time.time()
    m = func(A, num_bases=num_bases)
    m.factorize(show_progress=True, niter=niter)
    fro_norm = m.ferr[-1]/(A.shape[0] + A.shape[1])

    assert fro_norm < 0.1
    print 'Fro.: %d, elapsed %d' % (fro_norm, time.time() - stime)

    stime = time.time()
    m.factorize(show_progress=False, compute_h=False, niter=niter)
    m.factorize(show_progress=False, compute_w=False, niter=niter)
    m.factorize(show_progress=False, compute_err=False, niter=niter)
    m.factorize(show_progress=True, niter=20)

    print ' additional tests - elapsed:', time.time() - stime
