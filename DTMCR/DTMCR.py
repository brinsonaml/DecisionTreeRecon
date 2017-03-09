#!/home/xiaolin/anaconda2/bin/python2.7
# Implementation of Decision tree based 2d reconstruction approach
# Paper: R Bostanabad et al., Stochastic microstructure characterization and reconstruction via supervised learning
# Code implemented by Xiaolin Li
# For validating deep learning approach 
# Feb 13, 2017
# @Northwestern University

import argparse
import sys
import matplotlib.pyplot as plt
from utils import *
from sklearn import tree
from sklearn.model_selection import cross_val_score
import copy
from math import ceil
import cPickle as pickle
from numpy.random import binomial
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
def main(if_train, source_path, S=[2,10], L=100, filename='temp'):
	# if_train: if training is requested
	# source_path: image for training if if_train='y'
	#              pre-trained model if if_train='n'

	if if_train in ['Y', 'y']:
		# the training process is requested by the user
		# run training now
		img = loadimg(source_path)
		# convert image to data
		# select best neighbor size
		
		bestS = selectS(img, S)

		# train the decision tree classifier on full data
		clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, 
                min_samples_split=20, min_samples_leaf=20,
                min_weight_fraction_leaf=0.0, max_features=None, 
                random_state=10, max_leaf_nodes=None, class_weight=None, presort=False)
		dataX, dataY = img2data(img, bestS)
		#print "Number of Observations: ", dataX.shape[0]
		treeClassfier = clf.fit(dataX, dataY)
		recon = prepEdge(img, bestS, L)
		ret = reconstruct(recon, treeClassfier, bestS, L)
		
		vf = np.mean(img) # target volume
		current_vf = np.mean(img) # current vf for the reconstruction

		if abs(vf-current_vf) > 0.5:
			print "[Warining] VF is very different from the original microstructure"

		ret = matchVF(ret, vf) # adjust VF

		with open(str(filename)+'.pickle', 'w') as f:
			pickle.dump([ret, recon], f)

def selectS(img, S):
	# based on the image, select neighbor size S using cross-validation scores
	# input:
	# 	img: binary image
	#	S: a list of two integers [S1, S2] -- S1 is the lower bound of the neighbor size range
	#										  S2 is the upper bound of the neighbor size range
	# output:
	# 	bestS: the best neighbor size

	bestS = -1
	bestScore = -1
	for ss in range(S[0], S[1]+1):
		dataX, dataY = img2data(img, ss)
		print "Shape of Observations: ", dataX.shape
		clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, 
                min_samples_split=20, min_samples_leaf=20,
                min_weight_fraction_leaf=0.0, max_features=None, 
                random_state=10, max_leaf_nodes=None, class_weight=None, presort=False)

		score_list = cross_val_score(clf, dataX, dataY, cv=10)
		score = score_list.mean()
		print "When S="+ str(ss)+ ', the cv score is '+str(score)
		print "Last ten elements:", dataX[0:2,-10:]

		if score > bestScore:
			bestScore = score
			bestS = ss
		print "The best neighbor size so far is: " + str(bestS) 
	return bestS

def prepEdge(img, S, L):
	# prep the boarder of the reconstruction image before conducting recon
	# input:
	# 	img: the original image
	#	S: size of the neighborhood selected previously
	# 	L: size of the reconstruction image
	# output:
	# 	recon: intialization with cropped edge 
	
	x = 0
	y = 0
	L1, L2 = np.array(img).shape
	n = int((L + 2*S) / L1) + 1
	m = int((L + 4*S) / L2) + 1
	# create a large canvas and the crop it to the recon's size
	canvas = np.tile(img, (n, m))
	temp = copy.deepcopy(canvas) 
	# canvas size should be 2S+L, 4S+L
	recon = copy.deepcopy(temp[0 : 2 * S + L, 0 : 4 * S + L])
	# clear the central area
	recon[S : (2*S+L), S : (3*S+L)] = 0

	return recon

def reconstruct(recon, treeClf, S, L):
	# decision tree based reconstruction via raster-scanning
	# input:
	# 	recon: initialization of the reconstruction (more specifically, the boarder has been initialized with cropped original images)
	# 	treeClf: tree classifier trained on data
	# 	S: size of neighborhood
	#	L: size of the reconstruction image

	# dimension check
	assert recon.shape == (2*S + L, 4*S + L)
	# since the edge has already been initialized, the first point to predict is (S,S)
	# the last point to predict is 2*S+L-1, 3*S+L-1
	for i in range(S, 2*S+L):
		for j in range(S, 3*S+L):
			# for each point to predict, the first thing is to synthesize input vector X
			X = vectorizeNeighbor(recon, i, j, S)
			p = treeClf.predict_proba(X)[0][1]
			y = binomial(1, p)
			recon[i][j] = y
	#return recon
	return crop(recon, 2*S, 2*S, 2*S+L-1, 2*S+L-1)
 








if __name__=='__main__':
	# define the parser
	parser = argparse.ArgumentParser(description="Decision Tree based MCR")
	parser.add_argument('train', choices=['y', 'n', 'Y', 'N'], help='[y/n] training DT model again?')
	parser.add_argument('source', type=str, help='path of the image/pre-trained model.')
	parser.add_argument('lb', type=int, help='The size of the neighborhood (lower bound)')
	parser.add_argument('ub', type=int, help='The size of the neighborhood (upper bound)')
	parser.add_argument('--L', type=int, help='The size of the reconstruction image')
	parser.add_argument('--f', type=str, help='The name of the output file, default is temp')
	# parse the args
	args = parser.parse_args()
	if_train = args.train
	source_path = args.source
	S = [args.lb, args.ub]
	L = int(args.L)
	f = args.f
	main(if_train, source_path, S, L, f)