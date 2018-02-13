import os
import sys

import random

import tensorflow as tf
import numpy as np
import pandas as pd


from tqdm import tqdm

from skimage.transform import resize
from skimage.io import imread, imshow, imread_collection, concatenate_images

import matplotlib.pyplot as plt

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3


PROJECT_PATH = os.getcwd()  
TRAIN_PATH = PROJECT_PATH + "\dataset\\train\\"  
TEST_PATH = PROJECT_PATH + "\dataset\\test\\"

def get_nuclei_dataset():
	#Let's read all the folder inside the dataset which contains images and mask.

	train_image_ids = next(os.walk(TRAIN_PATH))[1]
	test_image_ids = next(os.walk(TEST_PATH))[1]

	#print(train_image_ids)

	# The folder which contains images and masks are stored in the above two lists.

	# Now, Lets prepare the dataset for trining and testing 
	# First lets prepare X_train and Y_train (images and their masks)

	#This is for the images in the training set
	X_train = np.zeros((len(train_image_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8) # As the default is float double precision

	Y_train = np.zeros((len(train_image_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.uint8) # This is for the masks







	# Now time to fill above multidimensional np vectors with the training image and mask
	for n, id_t in tqdm(enumerate(train_image_ids), total = len(train_image_ids)):
		#print (n)
		#print (id_t)
		local_path = TRAIN_PATH + id_t
		img_path = local_path + "\\images\\" + id_t + ".png"  
		image_t = imread(img_path)[:,:,:IMG_CHANNELS]
		#print(image_t.shape)
		image_t = resize(image_t, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
		X_train[n] = image_t
		# Now here is the trick, for the mask images we have to combine them Mask images are black and white 
		label_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype= np.bool)
		for mask_images_file_name in next(os.walk(local_path + "\\masks\\"))[2]:
			""" This will return all the files inside the mask directory
			"""
			current_mask = imread(local_path + "\\masks\\" + mask_images_file_name) 
			current_mask = resize(current_mask, (IMG_HEIGHT, IMG_WIDTH), mode= 'constant', preserve_range=True)
			# As masks are black and white images we have to expand the dimension
			current_mask = np.expand_dims(current_mask, axis = -1) # from H X W to H X W X 1
			label_mask = np.maximum(label_mask , current_mask)
		Y_train[n] = label_mask

	# Check if training data looks all right
	ix = random.randint(0, len(train_image_ids))
	imshow(X_train[ix])
	plt.show()
	imshow(np.squeeze(Y_train[ix]))
	plt.show()


	print("Now for the test images:")

	X_test = np.zeros((len(test_image_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8) # As the default is float double precision


	for n, id_t in tqdm(enumerate(test_image_ids), total = len(test_image_ids)):
		#print (n)
		#print (id_t)
		local_path = TEST_PATH + id_t
		img_path = local_path + "\\images\\" + id_t + ".png"  
		image_t = imread(img_path)[:,:,:IMG_CHANNELS]
		#print(image_t.shape)
		image_t = resize(image_t, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
		X_test[n] = image_t



	return X_train, Y_train, X_test