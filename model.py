import csv
import tensorflow 
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D
from keras.layers.convolutional import Conv2D

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)
        
#Generator, only using the center camera image to train the network
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.2
            for batch_sample in batch_samples:
                
                steering_center = float(batch_sample[3])
                angles.extend([steering_center]) 
                
                center_name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                
                image1 = cv2.imread(center_name)
                center_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB) 

                images.extend([center_image])
                

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
from sklearn.utils import shuffle
# Set the batch size
batch_size=128

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

#this is the same network as Nvidia's deep learning network
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(36,(5,5),activation="relu", strides=(2, 2)))
model.add(Conv2D(48,(5,5),activation="relu", strides=(2, 2)))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1)) #the finsl layer is set to 1 output, because this is a regression problem

model.compile(loss='mse', optimizer='adam') #using mse loss and adam optimizer

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1) #set 5 epochs
model.save('model.h5')
model.save_weights("weights.h5") #save the model's weight for fine tuning
### print the keys contained in the history object
print(history_object.history.keys())
