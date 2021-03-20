# libraries imports
import cv2
from glob import glob
import numpy as np
from sklearn.utils import shuffle
import pickle
import os
# Method to load Images
def labels_images():
	labels_of_images = []
	# getting images
	images = glob("gestures/*/*.jpg")
	# sorting the images
	images.sort()
	# for loop to load images from
	for image in images: 
        #print(image)
		label = image[image.find(os.sep)+1: image.rfind(os.sep)]
		# reading Images using OpenCv
		img = cv2.imread(image, 0)
		labels_of_images.append((np.array(img, dtype=np.uint8), int(label)))
	return labels_of_images
# Length of labels_of_images
labels_of_images = labels_images()
labels_of_images = shuffle(shuffle(shuffle(shuffle(labels_of_images))))
images, labels = zip(*labels_of_images)
print("Length of labels_of_images", len(labels_of_images))
# Length of images_train
images_train = images[:int(5/6*len(images))]
print("Length of train_images", len(images_train))
with open("train_images", "wb") as f:
	pickle.dump(images_train, f)
del images_train
# Length of labels_for_train
labels_for_train = labels[:int(5/6*len(labels))]

print("Length of train_labels", len(labels_for_train))
with open("train_labels", "wb") as f:
	pickle.dump(labels_for_train, f)
del labels_for_train
# Length of images_for_test
images_for_test = images[int(5/6*len(images)):int(11/12*len(images))]
print("Length of test_images", len(images_for_test))
with open("test_images", "wb") as f:
	pickle.dump(images_for_test, f)
del images_for_test
# Length of labels_for_test
labels_for_test = labels[int(5/6*len(labels)):int(11/12*len(images))]
print("Length of test_labels", len(labels_for_test))
with open("test_labels", "wb") as f:
	pickle.dump(labels_for_test, f)
del labels_for_test
# Length of images_for_test
val_images = images[int(11/12*len(images)):]
print("Length of test_images", len(val_images))
with open("val_images", "wb") as f:
	pickle.dump(val_images, f)
del val_images
# Length of val_of_labels
val_of_labels = labels[int(11/12*len(labels)):]
print("Length of val_labels", len(val_of_labels))
with open("val_labels", "wb") as f:
	pickle.dump(val_of_labels, f)
del val_of_labels