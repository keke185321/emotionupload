from constants import *
import cv2
import pandas as pd
import numpy as np
import glob
from PIL import Image
from os.path import join
import scipy.misc
CASC_PATH = './haarcascade_files/haarcascade_frontalface_default.xml'
SIZE_FACE = 48
EMOTIONS = ['angry', 'tired', 'anxious', 'happy',  'sad', 'neutral']
SAVE_DIRECTORY = './data/'
SAVE_MODEL_FILENAME = 'Gudi_model_100_epochs_20000_faces'
SAVE_DATASET_IMAGES_FILENAME = 'data_set_fer2013.npy'
SAVE_DATASET_LABELS_FILENAME = 'data_labels_fer2013.npy'
SAVE_DATASET_IMAGES_TEST_FILENAME = 'test_set_fer2013.npy'
SAVE_DATASET_LABELS_TEST_FILENAME = 'test_labels_fer2013.npy'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)



FILE_PATH = 'data/new.csv'
data = pd.read_csv(FILE_PATH)
def data_to_image(data):
    #print data
    #data_image = np.fromstring(str(data), dtype = np.uint8, sep = ' ').reshape((SIZE_FACE, SIZE_FACE))
    data_image = Image.fromarray(data).convert('RGB')
    data_image = np.array(data)[:, :, ::-1].copy() 
    #data_image = format_image(data_image)
    return data_image
def emotion_to_vec(x):
    d = np.zeros(len(EMOTIONS))
    d[x] = 1.0
    return d
def get_files(train,emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob('/home/kliao/Project/emotion-recognition-neural-networks/data/'+train+'/'+emotion+'/*')
    return files
train_data = []
train_labels = []
test_data = []
test_labels = []
j=0

for i in EMOTIONS:
	training=get_files('train',i)
	testing=get_files('test',i)
	for item in training:
		image = cv2.imread(item) #open image
		if image is not None:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			print 'train: ',np.shape(image)
		    	train_data.append(image) #append image array to training data list
		    	train_labels.append(emotion_to_vec(j))
	for item in testing:
		image = cv2.imread(item) #open image
		if image is not None:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			print 'test',np.shape(image)
            		test_data.append(image) #append image array to training data list
            		test_labels.append(emotion_to_vec(j))
	j+=1
print len(train_data),len(train_labels),len(test_data),len(test_labels)
    

#print "Total: " + str(len(images))
np.save('data/data_set_fer2013.npy', train_data)
np.save('data/data_labels_fer2013.npy', train_labels)
np.save('data/test_set_fer2013.npy', test_data)
np.save('data/test_labels_fer2013.npy', test_labels)

def load_from_save():
    images      = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_IMAGES_FILENAME))
    labels      = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_LABELS_FILENAME))
    images_test = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_IMAGES_TEST_FILENAME))
    labels_test = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_LABELS_TEST_FILENAME))
    print 'first: ',np.shape(images),np.shape(labels),  np.shape(images_test),np.shape(labels_test)

load_from_save()











