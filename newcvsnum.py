from constants import *
import cv2
import pandas as pd
import numpy as np
from PIL import Image
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

def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  gray_border = np.zeros((150, 150), np.uint8)
  gray_border[:,:] = 200
  gray_border[((150 / 2) - (SIZE_FACE/2)):((150/2)+(SIZE_FACE/2)), ((150/2)-(SIZE_FACE/2)):((150/2)+(SIZE_FACE/2))] = image
  image = gray_border

  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )
  # None is we don't found an image
  if not len(faces) > 0:
    #print "No hay caras"
    return None
  max_area_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
      max_area_face = face
  # Chop image to face
  face = max_area_face
  image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
  # Resize image to network size

  try:
    image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation = cv2.INTER_CUBIC) / 255.
  except Exception:
    print("[+] Problem during resize")
    return None
  print image.shape
  return image


def emotion_to_vec(x):
    d = np.zeros(len(EMOTIONS))
    d[x] = 1.0
    return d

def flip_image(image):
    return cv2.flip(image, 1)

def data_to_image(data):
    #print data
    data_image = np.fromstring(str(data), dtype = np.uint8, sep = ' ').reshape((SIZE_FACE, SIZE_FACE))
    data_image = Image.fromarray(data_image).convert('RGB')
    data_image = np.array(data_image)[:, :, ::-1].copy() 
    data_image = format_image(data_image)
    return data_image

FILE_PATH = 'data/new.csv'
data = pd.read_csv(FILE_PATH)

labels = []
images = []
index = 1
total = data.shape[0]
filenum=0
for index, row in data.iterrows():
    emotion = emotion_to_vec(row['emotion'])
    image = data_to_image(row['pixels'])
    
    if row['emotion']==0: 
	imtype='angry'
	print 'yes'
    elif row['emotion']==3:
	imtype='happy'
    elif row['emotion']==4:
	imtype='sad'
    elif row['emotion']==6:
	imtype='neutral'
    else: break
    if image is not None:
        labels.append(emotion)
        images.append(image)
	scipy.misc.toimage(image).save('data/'+imtype+'/%s.jpg' %(filenum))
	print 'filenum', filenum
	filenum+=1

    else:
        print "Error"
    index += 1
    print "Progreso: {}/{} {:.2f}%".format(index, total, index * 100.0 / total)
    

print "Total: " + str(len(images))
#np.save('data/data_kike.npy', images)
#np.save('data/labels_kike.npy', labels)
