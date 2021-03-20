import cv2
import numpy as np #used for working with arrays
import pickle   # library used for serializing and de-serializing object structures

def build_squares(img): #function for making small box on the screen 
	p1, p2, p3, p4 = 420, 140, 10, 10
	d = 10
	imageCrop = None
	crop = None
	for i in range(10):
		for j in range(5):
			if np.any(imageCrop == None):
				imageCrop = img[p2:p2+p4, p1:p1+p3] #cropping the image 
			else:
				imageCrop = np.hstack((imageCrop, img[p2:p2+p4, p1:p1+p3])) # hstack used to stack the sequence of input arrays horizontally
			cv2.rectangle(img, (p1,p2), (p1+p3, p2+p4), (0,255,0), 1) #creating a rectangle 
			p1+=p3+d
		if np.any(crop == None):
			crop = imageCrop
		else:
			crop = np.vstack((crop, imageCrop)) #vstack stacks arrays in sequence vertically (row wise)
		imageCrop = None
		p1 = 420
		p2+=p4+d
	return crop

def get_hand_hist(): #creating hand histogram
	cam = cv2.VideoCapture(1) #captures the video
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	flagC, flagS = False, False
	imageCrop = None
	while True:
		img = cam.read()[1]
		img = cv2.flip(img, 1)
		img = cv2.resize(img, (640, 480))
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		
		keypress = cv2.waitKey(1)
		if keypress == ord('c'):		
			hsvCrop = cv2.cvtColor(imageCrop, cv2.COLOR_BGR2HSV) #cvtColor method is used to convert an image from one color space to another
			flagC = True
			hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256]) # calcHist to find the histogram of the full image
			cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
		elif keypress == ord('s'):
			flagS = True	
			break
		if flagC:	
			dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1) #calcBackProject is a way of recording how well the pixels of a given image fit the distribution of pixels in a histogram model.
			disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
			cv2.filter2D(dst,-1,disc,dst)
			blur = cv2.GaussianBlur(dst, (11,11), 0)
			blur = cv2.medianBlur(blur, 15)
			ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #applying threshold
			thresh = cv2.merge((thresh,thresh,thresh)) # merge aligning the rows from each based on common attributes or columns
			cv2.imshow("Thresh", thresh)
		if not flagS:
			imageCrop = build_squares(img)
		cv2.imshow("Set hand histogram", img)
	cam.release()
	cv2.destroyAllWindows()
	with open("hist", "wb") as f:
		pickle.dump(hist, f) #method is used when the Python objects have to be stored in a file


get_hand_hist()
