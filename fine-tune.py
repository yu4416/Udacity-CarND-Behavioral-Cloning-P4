import csv
import tensorflow 
import cv2
import sklearn
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D
from keras.layers.convolutional import Conv2D

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)
        
#Generator, using left camera image and right camera image to train the model
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
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                flip_left_steering = steering_left * (-1) #times -1 to accomodate the filped camera iamge
                flip_right_steering = steering_right * (-1)
                angles.extend([steering_left, steering_right, flip_left_steering, flip_right_steering])
                
                left_name = 'data/IMG/'+batch_sample[1].split('/')[-1]
                right_name = 'data/IMG/'+batch_sample[2].split('/')[-1]
                
                #flip left camera image and right camera image to create more training data
                image2 = cv2.imread(left_name)
                left_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                flip_left = cv2.flip(left_image,1)
                
                image3 = cv2.imread(right_name)
                right_image = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
                flip_right = cv2.flip(right_image,1)
                images.extend([left_image, right_image, flip_left, flip_right])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
from sklearn.utils import shuffle
# Set our batch size
batch_size=128

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

pretrained_model = Sequential()
pretrained_model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
pretrained_model.add(Cropping2D(cropping=((70,25),(0,0))))
pretrained_model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
pretrained_model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
pretrained_model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
pretrained_model.add(Convolution2D(64,3,3,activation="relu"))
pretrained_model.add(Convolution2D(64,3,3,activation="relu"))
pretrained_model.add(Flatten())
pretrained_model.add(Dense(100))
pretrained_model.add(Dense(50))
pretrained_model.add(Dense(10))
pretrained_model.add(Dense(1))

extracted_layers = pretrained_model.layers[:]
pretrained_model = Sequential(extracted_layers)
pretrained_model.load_weights("weights.h5")
pretrained_model.summary()
pretrained_model.compile(loss='mse', optimizer='adam')
history_object = pretrained_model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)
pretrained_model.save('model_new.h5')
pretrained_model.save_weights("weights_new.h5")
print(history_object.history.keys())