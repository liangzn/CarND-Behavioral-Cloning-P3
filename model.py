import pandas as pd
import cv2, csv, numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import Cropping2D, Lambda, Conv2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from scipy.stats import norm

SHOW_GRAPHS = False
EPOCHS = 2

def loadData(filename):
    df = pd.read_csv(filename, header=None, names = ["center", "left", "right", "steering", "throttle", 'break', 'speed'])
        
        if SHOW_GRAPHS: 
            graphData(df['steering'], 'Pre-normalizedSteering Angle')
        
        df = df.drop(df.query('steering==0').sample(frac=0.90).index)
        # df = df.drop(df.query('abs(steering)==1').sample(frac=0.75).index)
        
        if SHOW_GRAPHS: 
            graphData(df['steering'], 'Post-normalizedSteering Angle')
        
        print(df.describe())
        
        return df

def graphData(x, title):
    n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
        mu, sigma = norm.fit(x)
        y = mlab.normpdf(bins, mu, sigma)
        plt.plot(bins, y, 'r--')
        # plt.set_xlim([-1, 1])
        if title: plt.title(title)
        plt.show()

def processData(data):
    images = []
        measurements = []
        
        for idx, row in data.iterrows():
            source_path = row['center']
                filename = source_path.split('/')[-1]
                current_path = 'data/IMG/' + filename
                
                image = cv2.imread(current_path)
                measurement = float(row['steering'])
                if not idx % 2:
                    image = np.fliplr(image)
                        measurement = -measurement
                images.append(image)
            measurements.append(measurement)
    
        return np.array(images), np.array(measurements)

def crop(x):
    h, w, c = x.shape
        crop_size = int(h * 0.40)
        x = x[crop_size:h-crop_size, 0:w]
        return cv2.resize(x, (w, h), interpolation = cv2.INTER_AREA)

def preprocess(x):
    x = crop(x)
        h, w, c = x.shape
        
        pnt = int(w*0.15)
        src = np.array([
                        [pnt, 0],
                        [w-pnt, 0],
                        [w, h],
                        [0, h]
                        ], dtype = "float32")
            
                        dst = np.array([
                                        [0, 0],
                                        [w, 0],
                                        [w, h],
                                        [0, h]
                                        ], dtype = "float32")
                        
                        # calculate the perspective transform matrix and warp
                        # the perspective to grab the screen
                        M = cv2.getPerspectiveTransform(src, dst)
                        warp = cv2.warpPerspective(x, M, (w, h))
                        
        return warp

def tfResize(x):
    import tensorflow as tf # Keep this here so it's loaded in the model during drive
        return tf.image.resize_images(x, (66, 200))

def createModel():
    input_shape = (160,320,3)
        
        # set up cropping2D layer
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
        model.add(Cropping2D(cropping=((50,20), (0,0))))
        model.add(Lambda(tfResize))
        model.add(Conv2D(24, (5, 5), strides=(2,2), activation="relu"))
        model.add(Conv2D(36, (5, 5), strides=(2,2), activation="relu"))
        model.add(Dropout(0.2))
        model.add(Conv2D(48, (5, 5), strides=(2,2), activation="relu"))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dropout(0.2))
        model.add(Dense(50))
        model.add(Dropout(0.2))
        model.add(Dense(10))
        model.add(Dense(1))
        
        return model

def plot_model(model):
    try:
        from keras.utils.visualize_util import plot
        plot(model, show_shapes=True)
    except:
        pass

def run_via_generator(model, X_train, y_train):
    # define data preparation
    datagen = ImageDataGenerator(
                                 preprocessing_function=preprocess, 
                                 )
        
        # fit parameters from data
        datagen.fit(X_train)
        
        model.fit_generator(datagen.flow(X_train, y_train, batch_size=16),
                            steps_per_epoch=len(X_train), epochs=EPOCHS)
        return model

def run(model, X_train, y_train):
    model.compile(loss='mse', optimizer='adam')
        model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=EPOCHS)
        return model

def saveModel(model):
    print("Saving model...")
        model.save('model.h5')
        print("Done.");

if __name__ == '__main__':
    data = loadData('data/driving_log.csv')
        X_train, y_train = processData(data)
        model = createModel()
        model = run(model, X_train, y_train)
        saveModel(model)
