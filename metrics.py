# Based on: https://github.com/jinsc37/DIFRINT/blob/master/metrics.py

import os
import sys
import numpy as np
import cv2

def metrics(original_dir, pred_dir):
	image_paths = sorted([path for path in os.listdir(pred_dir) if path.endswith(".png")])

	# Create brute-force matcher object
	bf = cv2.BFMatcher()

	sift = cv2.SIFT_create()

	# Apply the homography transformation if we have enough good matches 
	MIN_MATCH_COUNT = 10 #10

	ratio = 0.7 #0.7
	thresh = 5.0 #5.0

	CR_seq = []
	DV_seq = []
	Pt = np.eye(3)
	P_seq = []

	for i in range(len(image_paths)):
		# Load the images in gray scale
		img1 = cv2.imread(original_dir + image_paths[i], 0)
		img1o = cv2.imread(pred_dir + image_paths[i], 0)

		# Detect the SIFT key points and compute the descriptors for the two images
		keyPoints1, descriptors1 = sift.detectAndCompute(img1, None)
		keyPoints1o, descriptors1o = sift.detectAndCompute(img1o, None)

		# Match the descriptors
		matches = bf.knnMatch(descriptors1, descriptors1o, k=2)

		# Select the good matches using the ratio test
		goodMatches = []

		for m, n in matches:
			if m.distance < ratio * n.distance:
				goodMatches.append(m)

		if len(goodMatches) > MIN_MATCH_COUNT:
			# Get the good key points positions
			sourcePoints = np.float32([ keyPoints1[m.queryIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
			destinationPoints = np.float32([ keyPoints1o[m.trainIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
			
			# Obtain the homography matrix
			M, _ = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=thresh)
			# M, _ = cv2.estimateAffine2D(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=thresh)
		#end

		# Obtain Scale, Translation, Rotation, Distortion value
		
		# WRONG This is not the scale
		# sx = M[0, 0]
		# sy = M[1, 1]
		# scaleRecovered = np.sqrt(sx*sy)

		# Based on https://math.stackexchange.com/questions/78137/decomposition-of-a-nonsquare-affine-matrix
		scaleRecovered = np.sqrt(M[0,1]**2 + M[0,0]**2)
		# scalexRecovered = np.sqrt(M[0,0]**2 + M[1,0]**2)
		# scaleyRecovered = np.sqrt(M[0,1]**2 + M[1,1]**2)
		# scaleRecovered = np.sqrt(scalexRecovered**2 + scaleyRecovered**2)


		# WRONG This is not the affine part right?
		w, _ = np.linalg.eig(M[0:2, 0:2])
		# w, _ = np.linalg.eig(M[0:2])
		w = np.sort(w)[::-1]
		DV = w[1]/w[0]
		#pdb.set_trace()

		CR_seq.append(1/scaleRecovered)
		DV_seq.append(DV)

		# For Stability score calculation
		if i+1 < len(image_paths):
			
			img2o = cv2.imread(pred_dir + image_paths[i+1], 0)

			keyPoints2o, descriptors2o = sift.detectAndCompute(img2o, None)
			matches = bf.knnMatch(descriptors1o, descriptors2o, k=2)
			goodMatches = []

			for m, n in matches:
				if m.distance < ratio * n.distance:
					goodMatches.append(m)

			if len(goodMatches) > MIN_MATCH_COUNT:
				# Get the good key points positions
				sourcePoints = np.float32([ keyPoints1o[m.queryIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
				destinationPoints = np.float32([ keyPoints2o[m.trainIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
				
				# Obtain the homography matrix
				M, _ = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=thresh)
				# print(M)
				# M, _ = cv2.estimateAffine2D(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=thresh)
			#end

			P_seq.append(np.matmul(Pt, M))
			Pt = np.matmul(Pt, M)
		sys.stdout.write('\rFrame: ' + str(i) + '/' + str(len(image_paths)))
		sys.stdout.flush()
		#end
	#end

	# Make 1D temporal signals
	P_seq_t = []
	P_seq_r = []
	
	#pdb.set_trace()
	for Mp in P_seq:
		#w, _ = np.linalg.eig(Mp[0:2,0:2])
		#w = np.sort(w)[::-1]
		#DV = w[1]/w[0]
		transRecovered = np.sqrt(Mp[0, 2]**2 + Mp[1, 2]**2)
		# Based on https://math.stackexchange.com/questions/78137/decomposition-of-a-nonsquare-affine-matrix
		thetaRecovered = np.arctan2(Mp[1, 0], Mp[0, 0]) * 180 / np.pi
		#thetaRecovered = DV
		P_seq_t.append(transRecovered)
		P_seq_r.append(thetaRecovered)

	# FFT
	fft_t = np.fft.fft(P_seq_t)
	fft_r = np.fft.fft(P_seq_r)
	# WRONG What is this for?
	fft_t = np.abs(fft_t)**2  
	fft_r = np.abs(fft_r)**2
	
	#freq = np.fft.fftfreq(len(P_seq_t))
	#plt.plot(freq, abs(fft_r)**2)
	#plt.show()
	#print(abs(fft_r)**2)
	#print(freq)
	fft_t = np.delete(fft_t, 0)
	fft_r = np.delete(fft_r, 0)
	fft_t = fft_t[:len(fft_t)//2]
	fft_r = fft_r[:len(fft_r)//2]

	SS_t = np.sum(fft_t[:5])/np.sum(fft_t)
	SS_r = np.sum(fft_r[:5])/np.sum(fft_r)

	# Print results
	# print('\n***Last H:')
	# print(M)
	print('\n')
	print('***Cropping ratio (Avg, Min):')
	print( str.format('{0:.4f}', np.min([np.mean(CR_seq), 1])) +' | '+ str.format('{0:.4f}', np.min([np.min(CR_seq), 1])) )
	print('***Distortion value:')
	print(str.format('{0:.4f}', np.absolute(np.min(DV_seq))) )
	print('***Stability Score (Avg, Trans, Rot):')
	print(str.format('{0:.4f}',  (SS_t+SS_r)/2) +' | '+ str.format('{0:.4f}', SS_t) +' | '+ str.format('{0:.4f}', SS_r) )

if __name__ == '__main__':
	metrics(in_src='./data/Stab_te_reg/07/', out_src='./output/OurStabReg2/07/')


