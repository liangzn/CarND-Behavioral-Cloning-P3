
# coding: utf-8

# In[ ]:

## Learning: Important to cast to float. For example: float(line[3])
## Learning: left_angle = float(batch_sample[3])+steering_ang_correction .. for left image we add not subtract!!!
## since it is a correction


# ### TUNING PARAMETERS

# In[ ]:

# Steering Angle Offset
# Note: Steering Angle is normalized to -1 and 1
steering_ang_correction = 2
zero_angle_keep = 0.8


# ### Python Imports

# In[ ]:

import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sklearn
import pandas
from sklearn.utils import shuffle
import csv
get_ipython().magic('matplotlib inline')


# ### Read CSV

# In[ ]:

csv_headers = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
data = pd.read_csv('CarSim_data/driving_log.csv', names=csv_headers)


# ### Image Processing Functions

# In[ ]:

def i_crop(I):
    return I[55:135,:]

def i_resize(I):
    return cv2.resize(I,(64, 64),interpolation=cv2.INTER_AREA)

def i_flip(I, steering):
    return cv2.flip(I,1), -steering

def i_jitter(I, steering):
    I = cv2.cvtColor(I, cv2.COLOR_RGB2HSV)
    I[:,:,2] = I[:,:,2]+(np.random.uniform(-20,20))
    return cv2.cvtColor(I, cv2.COLOR_HSV2RGB), steering


# ### Image Procesing Playground
# Used to test basic image processing on a single image

# In[ ]:

# II = cv2.imread('./CarSim_data/IMG/left_2017_03_07_19_43_54_867.jpg')
# II = cv2.cvtColor(II,cv2.COLOR_BGR2RGB)
# plt.imshow(II)


# ### Read CSV
# ### Discard zero steering angles

# In[ ]:

samples = []
hist_angle = []
with open('./CarSim_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if float(line[3]) == 0:
            # Remove zero steering angle randomly
            if np.random.random() < zero_angle_keep:
                hist_angle.append(float(line[3]))
                samples.append(line)
        else:
            hist_angle.append(float(line[3]))
            samples.append(line)


# In[ ]:

plt.hist(hist_angle, bins = 40);
plt.xlabel('Steering Angles')
plt.ylabel('Frequency')
plt.title('Steering Angle Histogram')


# ### Split dataset into Training and Validation

# In[ ]:

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# In[ ]:

print(len(train_samples))


# ## Keras Model

# In[ ]:

from keras.models import Sequential
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from keras.optimizers import Adam
#from keras.layers.normalization import BatchNormalization


# In[ ]:

# from keras.utils.visualize_util import plot


# In[ ]:

ch, row, col = 3,64,64


# In[ ]:

model = Sequential()

# Normalization
model.add(Lambda(lambda x: x/127.5-1., input_shape=(col, row, ch), output_shape=(col, row, ch)))

# Convolution Layers
model.add(Convolution2D(24,5,5, init='glorot_uniform', subsample=(2, 2), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))

model.add(Convolution2D(36,5,5, init='glorot_uniform', subsample=(2, 2), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))

model.add(Convolution2D(48,5,5, init='glorot_uniform', subsample=(2, 2), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))

model.add(Convolution2D(64,3,3, init='glorot_uniform', border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))

model.add(Convolution2D(64,3,3, init='glorot_uniform', border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(1164))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(1))


# In[ ]:

#print(model.summary())


# In[ ]:

# Configuration
model.compile(loss='mse', optimizer='adam')


# ### Generator

# In[ ]:

def myGenerator(samples, batch_size):
    num_samples = len(samples)
    while 1: # loop forever so generator never terminates
        shuffle(samples)
        # for logging
        batch_num_idx = 1
        for offset in range(0, num_samples, batch_size):
            
            #             print('Batch Number: ', batch_num_idx, ' End')
            #             print(' ')
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            
            for batch_sample in batch_samples:
                
                # Center Image ===================
                name = batch_sample[0].strip()
                center_image = i_resize(i_crop(cv2.imread(name)))
                center_image = cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                center_image_flip, center_angle_flip = i_flip(center_image, center_angle)
                images.append(center_image_flip)
                angles.append(center_angle_flip)
                
                center_image_jitter, center_angle_jitter = i_jitter(center_image, center_angle)
                images.append(center_image_jitter)
                angles.append(center_angle_jitter)
                
                center_image_flip_jitter, center_angle_flip_jitter = i_jitter(center_image_flip, center_angle_flip)
                images.append(center_image_flip_jitter)
                angles.append(center_angle_flip_jitter)
                
                
                
                # Left Image =====================
                name = batch_sample[1].strip()
                left_image = i_resize(i_crop(cv2.imread(name)))
                left_image = cv2.cvtColor(left_image,cv2.COLOR_BGR2RGB)
                left_angle = float(batch_sample[3])+steering_ang_correction
                images.append(left_image)
                angles.append(left_angle)
                
                left_image_flip, left_angle_flip = i_flip(left_image, left_angle)
                images.append(left_image_flip)
                angles.append(left_angle_flip)
                
                left_image_jitter, left_angle_jitter = i_jitter(left_image, left_angle)
                images.append(left_image_jitter)
                angles.append(left_angle_jitter)
                
                left_image_flip_jitter, left_angle_flip_jitter = i_jitter(left_image_flip, left_angle_flip)
                images.append(left_image_flip_jitter)
                angles.append(left_angle_flip_jitter)
                
                # Right Image
                name = batch_sample[2].strip()
                right_image = i_resize(i_crop(cv2.imread(name)))
                right_image = cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB)
                right_angle = float(batch_sample[3])-steering_ang_correction
                images.append(right_image)
                angles.append(right_angle)
                
                right_image_flip, right_angle_flip = i_flip(right_image, right_angle)
                images.append(right_image_flip)
                angles.append(right_angle_flip)
                
                right_image_jitter, right_angle_jitter = i_jitter(right_image, right_angle)
                images.append(right_image_jitter)
                angles.append(right_angle_jitter)
                
                right_image_flip_jitter, right_angle_flip_jitter = i_jitter(right_image_flip, right_angle_flip)
                images.append(right_image_flip_jitter)
                angles.append(right_angle_flip_jitter)
            
            X_train = np.array(images)
            #             print('X_train Shape')
            #             print(X_train.shape)
            #             print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            #             print(' ')
            y_train = np.array(angles)
            
            # for logging
            batch_num_idx = batch_num_idx+1
        
        yield shuffle(X_train, y_train)


# In[ ]:

# compile and train the model using generator function
train_generator = myGenerator(train_samples, batch_size=256)
validation_generator = myGenerator(validation_samples, batch_size=256)


# In[ ]:

model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)


# ### Save Model

# In[ ]:

import json

model_json = model.to_json()
with open ('model.json', 'w') as f:
    json.dump(model_json, f, indent=4, sort_keys=True, separators=(',', ':'))

# model.save_weights will only save the weights
model.save('model.h5')
print("Model Saved")


# ---
# # End of CarND-Behavioral-Cloning-P3

