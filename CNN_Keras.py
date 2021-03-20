import numpy as np
import pickle
import cv2, os
from glob import glob
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
from keras.utils import plot_model
K.set_image_dim_ordering('tf')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
	img = cv2.imread('gestures/1/100.jpg', 0) #reading any image in the dataset.
	return img.shape #returning image shape

def get_num_of_classes():
	return len(glob('gestures/*')) #sub-directories in gestures directory are no. of classes

image_x, image_y = get_image_size() #diving length and width of image size to x and y.

def cnn_model(): #function to create a cnn model to classify images.
    num_of_classes = get_num_of_classes() #variable which has total no of classes.
    model = Sequential() #allows you to easily stack sequential layers (and even recurrent layers) of the network in order from input to output.
    model.add(Conv2D(16, (2,2), input_shape=(image_x, image_y, 1), activation='relu')) #convolutional layer, which applies convolutional mask(filter) to input image and summarizes the presence of detected features in the input.
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')) #down sample the detection of features in feature maps.
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten()) #Flatten pixels of image 2d to 1d.
    model.add(Dense(128, activation='relu')) #deeply connected layer each neuron in the dense layer receives input from all neurons of its previous layer.performs vector multiplication
    model.add(Dropout(0.2)) #reduce overfitting dropping inputs sets 20 percent inputs to 0.
    model.add(Dense(num_of_classes, activation='softmax')) #Softmax converts a real vector to a vector of categorical probabilities.The elements of the output vector are in range (0, 1) and sum to 1.
    sgd = optimizers.SGD(lr=1e-2) #SGD solved the Gradient Descent problem by using only single records to updates parameters.
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) #compile model.
    filepath="cnn_model_keras2.h5" #file to save as.
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    checkpoint2 = keras.callbacks.EarlyStopping(patience=2) #to stop training if accuracy differs with .02 difference in epochs
    callbacks_list = [checkpoint1,checkpoint2]
    plot_model(model, to_file='model.png', show_shapes=True) #model structure plotting
    return model, callbacks_list #returning model

def train():
	with open("train_images", "rb") as f: #opening file of train_images which got with load images file
		train_images = np.array(pickle.load(f))
	with open("train_labels", "rb") as f: #opening file of train_labels which got with load images file
		train_labels = np.array(pickle.load(f), dtype=np.int32)

	with open("val_images", "rb") as f: #opening file of val_images which got with load images file
		val_images = np.array(pickle.load(f))
	with open("val_labels", "rb") as f: #opening file of val_labels which got with load images file
		val_labels = np.array(pickle.load(f), dtype=np.int32)

	train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1)) #numpy funct used reshape to adjust shapes of all images
	val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
	train_labels = np_utils.to_categorical(train_labels)
	val_labels = np_utils.to_categorical(val_labels)

	print(val_labels.shape)

	model, callbacks_list = cnn_model()
	model.summary()
	model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=20, batch_size=500, callbacks=callbacks_list)
	scores = model.evaluate(val_images, val_labels, verbose=0)
	print("CNN Error: %.2f%%" % (100-scores[1]*100))
	model.save('cnn_model_keras2.h5')

train()
K.clear_session();