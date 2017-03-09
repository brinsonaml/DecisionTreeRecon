#utility functions
import sys
import scipy.io as sio
import numpy as np
import copy
from numpy.random import permutation

def loadimg(PATH):
	if PATH[-4:] == '.mat':
		data = sio.loadmat(PATH)
		img = data['img']
	else:
		sys.exit('[Error] Failed to identify image type')

	return img

def img2data(img, S):
	#convert image into dataset
	#input
	#	img: binary image
	#	S: the size of the neighbor hood
	#output
	#	trainX: training set -- predictors
	# 	trainY: training set -- responses
	
	img = np.array(img)
	
	L1, L2 = img.shape

	# the first response point's coordinate is [S-1, S] (edge length is S)
	trainX = []
	trainY = []
	# pre-allocate the memory space for X to reduce the run time
	X = np.ones( 2 * S * ( S - 1) + S) * -1
	for row in range(S-1, L1):
		for col in range(S, L2 - S + 1):
			# missing data is not accepted in creating the data set
			y = img[row][col]
			idx = 0
			left = col - S
			right = col + S - 1

			for i in range(row - S + 1, row):
				for j in range(left, right + 1):
					X[idx] = img[i][j]
					idx += 1

			for j in range(left, col):
				X[idx] = img[row][j]
				idx += 1
			
			trainX.append(copy.copy(X))
			trainY.append(y)

	return np.array(trainX), np.array(trainY)

def vectorizeNeighbor(img, i, j, S):
	# given the coordinate of a point (i,j), pick its neighbor area and 
	# vectorize them into input X
	# input:
	# 	img: image
	#	i: x coordinate
	#	j: y coordinate
	#	S: neighbor size
	X = []
	X = np.ones(2*S*(S-1)+S) * -1
	idx = 0
	for m in range(i - S + 1, i):
		for n in range(j-S, j+S):
			X[idx] = img[m][n]
			idx += 1
	for n in range(j-S, j):
		X[idx] = img[i][n]
		idx += 1
	assert idx == len(X)
	return X

def crop(img, x1, y1, x2, y2):
	L1, L2 = img.shape
	assert x1>=0
	assert y1>=0
	assert x2>=0
	assert y2>=0
	assert x1<L1
	assert x2<L1
	assert y1<L2
	assert y2<L2
	if x1>x2:
		x1, x2 = x2, x1
	if y1>y2:
		y1, y2 = y2, y1
	return img[x1:x2+1, y1:y2+1]

def findEdge1(img):
	# find the 1 edge (4-neighbor) from the binary image
	# input:
	# 	img: binary image
	L1, L2 = img.shape
	img1 = np.zeros([L1, L2])
	img2 = np.zeros([L1, L2])
	img3 = np.zeros([L1, L2])
	img4 = np.zeros([L1, L2])

	img1[:L1-1, :]=img[1:, :]
	img2[1:, :]=img[:L1-1, :]
	img3[:, :L2-1] = img[:, 1:]
	img4[:, 1:] = img[:, :L2-1]

	canvas = img1+img2+img3+img4-4*img
	ret = []
	for i in range(L1):
		for j in range(L2):
			if canvas[i][j] < 0:
				ret.append([i,j])
	return np.array(ret)

def findEdge0(img):
	# find the 0 edge (4-neighbor) from the binary image
	# input:
	# 	img: binary image
	L1, L2 = img.shape
	img1 = np.zeros([L1, L2])
	img2 = np.zeros([L1, L2])
	img3 = np.zeros([L1, L2])
	img4 = np.zeros([L1, L2])

	img1[:L1-1, :]=img[1:, :]
	img2[1:, :]=img[:L1-1, :]
	img3[:, :L2-1] = img[:, 1:]
	img4[:, 1:] = img[:, :L2-1]

	canvas = img1+img2+img3+img4-4*img
	ret = []
	for i in range(L1):
		for j in range(L2):
			if canvas[i][j] > 0:
				ret.append([i,j])
	return np.array(ret)

def matchVF(img, vf):
	# erosion/dilation of the current image to achieve target VF
	# input:
	# 	img: image whose VF needs to be adjusted
	#	vf: target volumn fraction
	currentVF = np.mean(img)
	L1, L2 = img.shape
	increment = 1./L1/L2
	if currentVF < vf:
		# dilation
		c_list = findEdge0(img)
	elif currentVF > vf:
		# erosion
		c_list = findEdge1(img)
	#print c_list
	n = int(abs(currentVF - vf) / increment)
	m = c_list.shape[0]

	while m<=n:
		# the number of boundary pixels is less than the number of pixels to be adjusted.
		# Then we need to change them all
		if currentVF < vf:
			for i in range(m):
				img[c_list[i,0], c_list[i,1]] = 1
			c_list = findEdge0(img)
		elif currentVF > vf:
			for i in range(m):
				img[c_list[i, 0], c_list[i,1]] = 0
			c_list = findEdge1(img)
		currentVF = np.mean(img)
		n = int(abs(currentVF - vf) / increment)
		m = c_list.shape[0]

	if n>0:
		# if there are still pixels that are expected to change
		c_list_len = c_list.shape[0]
		selected = permutation(c_list_len)[:n]
		for ele in selected:
			if currentVF < vf:
				img[c_list[ele, 0], c_list[ele, 1]] = 1
			else:
				img[c_list[ele, 0], c_list[ele, 1]] = 0

	return img










