import csv
import cv2
import numpy as np
from scipy import ndimage

lines = []
# Udacity Provided Training Data
#directory = '/opt/carnd_p3/data/'

# My training data
directory = '/root/Desktop/data/'

with open(directory + 'driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    #Skip the header row
    next(reader)
    for line in reader:
        lines.append(line)

images = []
measurements = []
corr_fact = 0.2

for line in lines:
    for i in range(3):
        source_path = line[i]
        tokens = source_path.split('/')
        filename = tokens[-1]
        local_path = directory + 'IMG/' + filename
        
        # Note that this takes in images in BGR format, but I need RGB format, either convert or use another alternative
        image = ndimage.imread(local_path)
        #image = cv2.imread(local_path)
        images.append(image)
        image_flipped = np.fliplr(image)
        images.append(image_flipped)
    
    measurement = float(line[3])
    #measurement_flipped = -measurement
    measurements.append(measurement)
    measurements.append(measurement + corr_fact)
    measurements.append(measurement - corr_fact)
    #measurements.append(measurement_flipped)
    
aug_images = []
aug_measurements = []
for img, meas in zip(images, measurements):
    aug_images.append(image)    
    image_flipped = np.fliplr(image)
    aug_images.append(image_flipped)
    
    aug_measurements.append(measurement)
    measurement_flipped = -1.0 * float(measurement)    
    aug_measurements.append(measurement_flipped)

# Compile arrays of training data
X_train = np.array(aug_images)
y_train = np.array(aug_measurements)

# Create model
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Normalize the images
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

# Crop the images to remove data from the image that can distract the network from predicting the lane
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# Add first convolutional layer
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

# Add second convolutional layer
model.add(Convolution2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))
          
# Add fully connected layer
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1)

model.save('model.h5')
print("Run the command: python drive.py model.h5")