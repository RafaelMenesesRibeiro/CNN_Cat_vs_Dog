import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm

#-------------------------------------------------------------------------------

	#MAIN STRUCTURES AND VARIABLES

#-------------------------------------------------------------------------------
TRAIN_DIR = '' #Directory where the images are.
IMG_SIZE = 50 #Resized image size (to feed as input).
LR = 1e-3 #Convolutional Neural Network Learning Rate.
#CNN model name. Used to save and load the model (so retraining isn't necessary).
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic')
TRAIN_DATA_NAME = 'dogvscatsTrainingData.npy'

#-------------------------------------------------------------------------------

	#FUNCTIONS

#-------------------------------------------------------------------------------
def getOneHotLabel(imageName):
	'''The names of the images of this dataset are as dog.93.png.
	It is divided into ['dog', '93', 'png'].
	word_label will be 'dog'.'''
	word_label = imageName.split('.')[-3]
	#Returns the appropriate One-Hot vector.
	labels = ['cat', 'dog']
	out = [0] * len(labels)
	out[labels.index(word_label)] = 1
	return out

def trainDataCreate():
	training_data = []
	for img in tqdm(os.listdir(TRAIN_DIR)):
		one_hot = getOneHotLabel(img) #Gets the One-Hot label for the image.
		path = os.path.join(TRAIN_DIR, img) #Gets the image's absolute path.
		data = cv2.resize(cv2.imread(path, 0), (IMG_SIZE, IMG_SIZE)) #Gets the image.
		#Adds the image's data and label to the training data.
		training_data.append([np.array(data), np.array(one_hot)])
	shuffle(training_data)
	#Saves the processed training data to save time when retraining (with the same data).
	np.save(TRAIN_DATA_NAME, training_data)
	return training_data

def modelCreate():
	convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
	convnet = conv_2d(convnet, 32, 2, activation='relu')
	convnet = max_pool_2d(convnet, 2)
	convnet = conv_2d(convnet, 64, 2, activation='relu')
	convnet = max_pool_2d(convnet, 2)
	convnet = conv_2d(convnet, 32, 2, activation='relu')
	convnet = max_pool_2d(convnet, 2)
	convnet = conv_2d(convnet, 64, 2, activation='relu')
	convnet = max_pool_2d(convnet, 2)
	convnet = conv_2d(convnet, 32, 2, activation='relu')
	convnet = max_pool_2d(convnet, 2)
	convnet = conv_2d(convnet, 64, 2, activation='relu')
	convnet = max_pool_2d(convnet, 2)
	convnet = fully_connected(convnet, 1024, activation='relu')
	convnet = dropout(convnet, 0.8)
	convnet = fully_connected(convnet, 2, activation='softmax')
	convnet = regression(convnet, optimizer='adam', learning_rate = LR, loss = 'categorical_crossentropy', name = 'targets')
	return tflearn.DNN(convnet, tensorboard_dir = 'log')

def modelFit(model):
	#Training Data
	train = train_data[:-500]
	X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
	Y = [i[1] for i in train]
	#Testing Data
	test = train_data[-500:]
	test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
	test_y = [i[1] for i in test]
	#Fits the model.
	model.fit(
		{'input' : X}, {'targets' : Y},
		n_epoch = 5, 
		validation_set = ({'input' : test_x}, {'targets' : test_y}), 
		snapshot_step = 500,
		show_metric = True,
		run_id = MODEL_NAME)
	model.save(MODEL_NAME) #Saves the model so retraining isn't necessary.
	return model

def queryImageShow(imagePath, imageLabel):
	cv2.imshow(['Cat', 'Dog'][np.argmax(imageLabel)], cv2.resize(cv2.imread(imagePath, 0), (300, 300)))
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#-------------------------------------------------------------------------------

	#CODE EXECUTION

#-------------------------------------------------------------------------------
print('Processing training data...')
train_data = trainDataCreate() #Creates the training data.
#train_data = np.load(TRAIN_DATA_NAME) #Loads the training data.
print('Done processing.\n')

print('Creating model...')
model = modelCreate() #Creates the model
print('Done creating model.\n')
#Loads the model if it exists.
if os.path.exists('{}.meta'.format(MODEL_NAME)):
	model.load(MODEL_NAME)
	print('Done loading model.\n')
#Trains the model if it doesn't exist.
else:
	print('Training model...')
	model = modelFit(model)
	print('Done training model.\n')

#Queries the user for the path of the image to categorize.
path = input('Absolute path to image:\n')
image2D = np.array(cv2.resize(cv2.imread(path, 0), (IMG_SIZE, IMG_SIZE)))
imageData = image2D.reshape(IMG_SIZE, IMG_SIZE, 1)
prediction = model.predict([imageData])[0]
print('Cat: ', prediction[0], '\tDog: ', prediction[1])
queryImageShow(path, prediction)