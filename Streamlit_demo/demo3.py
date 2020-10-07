#!/usr/bin/env python
# coding: utf-8

# In[2]:





# In[6]:





# In[16]:





# In[18]:





# In[21]:


from __future__ import division, print_function
# coding=utf-8
import streamlit as st
import h5py
from PIL import Image
import os
import numpy as np
import json
import predict3
from tensorflow.keras.models import load_model
import keras
from keras.models import Model
from keras.layers import Dense, Activation
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.optimizers import Adam


import argparse

import pandas as pd

from tqdm import tqdm


IM_WIDTH, IM_HEIGHT = 299, 299
NB_EPOCHS = 5
BATCH_SIZE = 32
FC_SIZE = 1024

# Get Inception model without final layer
base_model = InceptionV3(weights='imagenet', include_top=False)

# Add new fully connected layer to base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(FC_SIZE, activation='relu')(x)
predictions = Dense(38, activation='softmax')(x)
model = Model(base_model.input, predictions)
# Compile model
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#from tensorflow.keras.models import load_weights
# Model path
folder_path = "./models"
#model_name = "model_vgg16_2.hdf5"
#model_name = "inception_1.h5"
model_name = "inception_15ep.h5"
model_file = os.path.join(folder_path, model_name)

# Load your trained model
#model = load_model(model_file)
#import joblib
#model = joblib.load('./models/export_resnet34_model.pkl')
#import pickle
#pkl_filename = './models/export_resnet34_model.pkl'
#with open(pkl_filename, 'rb') as file:
	#model = pickle.load(file)
model.load_weights(model_file)
st.markdown("<h1 style='text-align: left; color: green;'>Welcome to your plant HealthScanner!</h1>", unsafe_allow_html=True)
st.write("")
image = Image.open('./models/logo_crop_2.png')
st.sidebar.image(image, use_column_width = True)
st.sidebar.title('Image diagnosis')
st.set_option('deprecation.showfileUploaderEncoding', False)
img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
	#image = np.array(Image.open(img_file_buffer))
	image = Image.open(img_file_buffer)
	st.image(image, caption='Uploaded Image.')
	processed_image = predict3.preprocess_image(img_file_buffer)
	prediction = predict3.model_predict(processed_image, model)
	st.write("### Diagnosis results:")
	res = '%s : %s' % (prediction[0][0], prediction[0][1])
	st.write(res)
	st.write("### Disease description:")
	descr = predict3.description(prediction)
	st.write(descr[0][1])

	st.write("### Treatment recommendations:")
	tx = predict3.treatment(prediction)
	st.write(tx[0][1])
else:
	# st.sidebar.success("Select an image above.")

	st.markdown(
		"""
		HealthScanner help you diagnose your lovely plants, and recommend proper treatments for you!

		Upload an image on the left to have quick diagnosis!




		"""
		)

image2 = Image.open('./models/img_crop.jpg')
st.image(image2, use_column_width = True)
