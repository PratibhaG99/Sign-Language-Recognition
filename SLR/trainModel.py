import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense

train_path=r"C:\Users\INTEL 2022\Desktop\gesture\Train"
valid_path=r"C:\Users\INTEL 2022\Desktop\gesture\Valid"
test_path=r"C:\Users\INTEL 2022\Desktop\gesture\Test"
train_batches = ImageDataGenerator().flow_from_directory(directory=train_path,
                                    target_size=(64,64),class_mode='categorical', batch_size=150,shuffle=True)
valid_batches = ImageDataGenerator().flow_from_directory(directory=valid_path,
                                    target_size=(64,64),class_mode='categorical', batch_size=50,shuffle=True)
test_batches = ImageDataGenerator().flow_from_directory(directory=test_path,
                                    target_size=(64,64),class_mode='categorical', batch_size=70,shuffle=True)

imgs, labels = next(valid_batches)

from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))
model.add(Dense(128,activation ="relu"))
model.add(Dense(12,activation ="softmax"))

model.summary()

model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
#early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
history2 = model.fit(train_batches, epochs=10, callbacks=[reduce_lr],validation_data = valid_batches, verbose=2)

model.save('Sample_Model.h5')


Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_4 (Conv2D)           (None, 62, 62, 32)        896       
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 31, 31, 32)       0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (None, 31, 31, 64)        18496     
                                                                 
 conv2d_6 (Conv2D)           (None, 31, 31, 64)        36928     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 15, 15, 64)       0         
 2D)                                                             
                                                                 
 conv2d_7 (Conv2D)           (None, 13, 13, 128)       73856     
                                                                 
 max_pooling2d_5 (MaxPooling  (None, 6, 6, 128)        0         
 2D)                                                             
                                                                 
 flatten_1 (Flatten)         (None, 4608)              0         
                                                                 
 dense_4 (Dense)             (None, 64)                294976    
                                                                 
 dense_5 (Dense)             (None, 128)               8320      
                                                                 
 dense_6 (Dense)             (None, 128)               16512     
                                                                 
 dense_7 (Dense)             (None, 12)                1548      
                                                                 
=================================================================
Total params: 451,532
Trainable params: 451,532
Non-trainable params: 0
_________________________________________________________________