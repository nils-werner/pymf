#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: dist.py 24 2010-09-01 07:51:05Z cthurau $
#$Author$
"""
PyMF several distance functions

	kl_divergence(): KL Divergence
	l1_distance(): L1 distance
	l2_distance(): L2 distance
	cosine_distance(): Cosine distance 
	pdist(): Pairwise distance computation
	vq(): Vector quantization
	
"""

__version__ = "$Revision$"

import numpy as np
import scipy.sparse

__all__ = ["kl_divergence", "l1_distance", "l2_distance", "cosine_distance",
		"vq", "pdist","sparse_graph_l2_distance"]

def kl_divergence(d, vec):	
	b = vec*(1/d)	
	b = np.where(b>0, np.log(b),0)	
	b = vec * b
	b = np.sum(b - vec + d, axis=0).reshape((-1,1))			
	return b
		
def l1_distance(d, vec):			
	ret_val = np.sum(np.abs(d - vec), axis=0)					
	ret_val = ret_val.reshape((-1,1))						
	return ret_val
	

def sparse_graph_l2_distance(d, vec):	
	ret_val = d.sum(axis=0) + vec.sum()
	tmp = (d.T * vec).todense()	
	ret_val = ret_val.T - (2.0*tmp)	
	ret_val = np.where(ret_val > 0, np.sqrt(ret_val), ret_val)
	return ret_val

def sparse_l2_distance(d, vec):
	ret_val = np.zeros((d.shape[1],1))		
	idx2 = vec.nonzero()[0]

	for i in range(d.shape[1]):
			idx1 = d[:,i:i+1].nonzero()[0]
			tmp = np.setxor1d(idx1, idx2)
			ret_val[i,0] = np.sqrt(len(tmp))				
	return ret_val
		
def l2_distance(d, vec):	
	if scipy.sparse.issparse(d):
		ret_val = sparse_l2_distance(d, vec)
	else:
		ret_val = np.sqrt(((d[:,:] - vec)**2).sum(axis=0))
			
	return ret_val.reshape((-1,1))		
		
def cosine_distance(d, vec):
	tmp = np.dot(np.transpose(d), vec)
	a = np.sqrt(np.sum(d**2, axis=0))
	b = np.sqrt(np.sum(vec**2))
	k = (a*b).reshape(-1,1) + 10**-9
			
	# compute distance
	ret_val = 1.0 - tmp/k

	# check for NaN values			
	ret_val[np.isnan(ret_val)] = 0
		
	return ret_val

def pdist(A, B, metric='l2' ):
	# compute pairwise distance between a data matrix A (d x n) and B (d x m).
	# Returns a distance matrix d (n x m).
	d = np.zeros((A.shape[1], B.shape[1]))
	if A.shape[1] <= B.shape[1]:
		for aidx in xrange(A.shape[1]):
			d[aidx:aidx+1,:] = l2_distance(B[:,:], A[:,aidx:aidx+1]).reshape((1,-1))
	else:
		for bidx in xrange(B.shape[1]):
			d[:, bidx:bidx+1] = l2_distance(A[:,:], B[:,bidx:bidx+1]).reshape((-1,1))
	
	return d

def vq(A, B):
	# assigns data samples in B to cluster centers A and
	# returns an index list [assume n column vectors, d x n]
	assigned = np.argmin(pdist(A,B), axis=0)
	return assigned