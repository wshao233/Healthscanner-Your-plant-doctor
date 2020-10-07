"""Train CNN model based on InceptionV3 network."""

import keras
from keras.models import Model
from keras.layers import Dense, Activation
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.optimizers import Adam

import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm


IM_WIDTH, IM_HEIGHT = 299, 299
NB_EPOCHS = 15
BATCH_SIZE = 32
FC_SIZE = 1024

TRAIN_DIR ="./train/"
VAL_DIR = "./test/"
EXPORT_DIR = "./models/"

# Get image classes
classes = os.listdir(TRAIN_DIR)
num_classes = len(classes)

# Get path and label for each image
db = []
for label, class_name in enumerate(classes):

    # Train
    path = TRAIN_DIR + class_name
    for file in os.listdir(path):
        db.append(['{}/{}'.format(class_name, file), label, class_name, 1])

    # Validation
    path = VAL_DIR + class_name
    for file in os.listdir(path):
        db.append(['{}/{}'.format(class_name, file), label, class_name, 0])

db = pd.DataFrame(db, columns=['file', 'label', 'class_name', 'train_ind'])

num_train_samples = db.train_ind.sum()
num_val_samples = len(db) - num_train_samples

# Specify data generator inputs
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=BATCH_SIZE,
)
validation_generator = test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=BATCH_SIZE,
)

# Get Inception model without final layer
base_model = InceptionV3(weights='imagenet', include_top=False)

# Add new fully connected layer to base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(FC_SIZE, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(base_model.input, predictions)
# Compile model
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Set up callbacks


# Fit transfer learning model
history = model.fit_generator(
    train_generator,
    epochs=NB_EPOCHS,
    steps_per_epoch=500, #num_train_samples / BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=200) #num_val_samples / BATCH_SIZE,
model.save( "inception_15ep.h5" )
