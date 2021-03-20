from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=True): #creating a confusion_matrix 

    """
    given a sklearn confusion matrix (cm)

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Use
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    """
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(20, 20)) #used to create a new figure
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title) #method returns a string where the first character in every word is upper case
    plt.colorbar()  # colorbar to a plot indicating the color scale

    if target_names is not None:
        tick_marks = np.arange(len(target_names)) # arange one of the array creation routines based on numerical ranges
        plt.xticks(tick_marks, target_names, rotation=45) # used to get and set the current tick locations and labels of the x-axis
        plt.yticks(tick_marks, target_names)  #used to get and set the current tick locations and labels of the y-axis

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # used to cast a pandas object to a specified dtype


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout() #used to automatically adjust subplot parameters to give specified padding
    plt.ylabel('True label') 
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('confusion_matrix.png')


image_x, image_y = 50, 50
with open("test_images", "rb") as f:
	test_images = np.array(pickle.load(f))
with open("test_labels", "rb") as f:
	test_labels = np.array(pickle.load(f), dtype=np.int32)
test_images = np.reshape(test_images, (test_images.shape[0], image_x, image_y, 1)) # used to give a new shape to an array without changing its data


model = load_model('cnn_model_keras2.h5') #used to load a model 
pred_labels = []

start_time = time.time()
pred_probabs = model.predict(test_images) #enables us to predict the labels of the data values on the basis of the trained model
end_time = time.time() 
pred_time = end_time-start_time
avg_pred_time = pred_time/test_images.shape[0]
print("Time taken to predict %d test images is %ds" %(test_images.shape[0], pred_time))
print('Average prediction time: %fs' % (avg_pred_time))

for pred_probab in pred_probabs:
	pred_labels.append(list(pred_probab).index(max(pred_probab)))

cm = confusion_matrix(test_labels, np.array(pred_labels))
classification_report = classification_report(test_labels, np.array(pred_labels))
plot_confusion_matrix(cm, range(44), normalize=False)