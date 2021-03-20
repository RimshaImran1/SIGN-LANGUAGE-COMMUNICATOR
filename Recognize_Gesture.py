import cv2, pickle
import numpy as np
import tensorflow as tf
import os
import sqlite3
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
prediction = None
model = load_model('cnn_model_keras2.h5') #load cnn trained model file.

def get_image_size(): #select any img from directory and return size
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

image_x, image_y = get_image_size() #divide x and y length width of img shape

def keras_process_image(img): #process img
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image): #predict probability
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class #max probability class it belongs to

def get_pred_text_from_db(pred_class): #check the word against that img from gestures db.
	conn = sqlite3.connect("gesture_db.db")
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class) #select that word and store in cmd.
	cursor = conn.execute(cmd)
	for row in cursor: #run-time output
		return row[0]

def show_text(text, num_of_words): #show text in blackboard

	list_words = text.split(" ")
	length = len(list_words)
	show_text = []
	b_index = 0
	e_index = num_of_words
	while length > 0:
		part = ""
		for word in list_words[b_index:e_index]:
			part = part + " " + word
		show_text.append(part)
		b_index += num_of_words
		e_index += num_of_words
		length -= num_of_words
	return show_text

def put_text_in_blackboard(blackboard, word_text): #putting text in blackboard
	y = 200
	for text in word_text:
		cv2.putText(blackboard, text, (4, y), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		y += 50

def get_hand_hist(): #load hist file and return histogram
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

def recognize():
	global prediction 
	cam = cv2.VideoCapture(1) #video capture
	if cam.read()[0] == False: #read input 
		cam = cv2.VideoCapture(0) 
	hist = get_hand_hist() #hist of hand
	x, y, w, h = 300, 100, 300, 300 #dimensions
	while True: #continuous runtime output
		text = ""
		img = cam.read()[1] #read and store input
		img = cv2.flip(img, 1) #flip img
		img = cv2.resize(img, (640, 480)) #resize every img
		imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
		disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
		cv2.filter2D(dst,-1,disc,dst)
		blur = cv2.GaussianBlur(dst, (11,11), 0)
		blur = cv2.medianBlur(blur, 15)
		thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] #applying threshold
		thresh = cv2.merge((thresh,thresh,thresh))
		thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
		thresh = thresh[y:y+h, x:x+w]
		(openCV_ver,_,__) = cv2.__version__.split(".")
		if openCV_ver=='3':
			contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]#applying contours
		elif openCV_ver=='4':
			contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				x1, y1, w1, h1 = cv2.boundingRect(contour)
				save_img = thresh[y1:y1+h1, x1:x1+w1]
				
				if w1 > h1:
					save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
				elif h1 > w1:
					save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
				
				pred_probab, pred_class = keras_predict(model, save_img)
				
				if pred_probab*100 > 80:
					text = get_pred_text_from_db(pred_class)
					print(text)# printing text on console
		blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
		word_text = show_text(text, 2) #show on blackboard
		put_text_in_blackboard(blackboard, word_text) #putting words in blackboard
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2) #Green Square
		res = np.hstack((img, blackboard))
		cv2.imshow("Recognizing gesture", res) #show window of recognize gesture
		cv2.imshow("thresh", thresh) #show threshold
		if cv2.waitKey(1) == ord('q'):
			break

keras_predict(model, np.zeros((50, 50), dtype=np.uint8))		
recognize()
