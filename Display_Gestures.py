#import of libraries
import cv2, os, random
import numpy as np
# geting size of image
def get_image_size():
	# reading image
	img = cv2.imread('gestures/0/100.jpg', 0)
	# images return
	return img.shape
# storing gestures
gestures = os.listdir('gestures/') #directory
gestures.sort(key = int)
# start index
start_index = 0
# end index
last_index = 5
# geting size of images

x_image, image_y = get_image_size()

if len(gestures)%5 != 0:
	rows = int(len(gestures)/5)+1
else:
	rows = int(len(gestures)/5)
# displaying images
full_image = None
# for loop to display All the gestures
for i in range(rows):
	col_img = None
	for j in range(start_index, last_index):
		#open image path and store in array
		img_path = "gestures/%s/%d.jpg" % (j, random.randint(1, 1200))
		#image read
		img = cv2.imread(img_path, 0)
		if np.any(img == None):
			# initialize Mat and fill with Zeros
			img = np.zeros((image_y, x_image), dtype = np.uint8)
		if np.any(col_img == None):
			col_img = img
		else:
			col_img = np.hstack((col_img, img)) # 1d stack array in numpy for column-wise

	start_index += 5
	last_index += 5
	if np.any(full_image == None):
		full_image = col_img
	else:
		full_image = np.vstack((full_image, col_img)) # 1d stack array in numpy for row-wise

# showing images
cv2.imshow("gestures", full_image)
# writing images

cv2.imwrite('full_image.jpg', full_image)
cv2.waitKey(0)