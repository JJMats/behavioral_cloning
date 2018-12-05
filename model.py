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
    print("Getting image files...")
    reader = csv.reader(csvFile)
    #Skip the header row
    next(reader)
    for line in reader:
        lines.append(line)

images = []
measurements = []
corr_fact = 0.2

for line in lines:
    source_path = line[0]
    tokens = source_path.split('/')
    filename = tokens[-1]
    local_path = directory + 'IMG/' + filename
        
    image = ndimage.imread(local_path)
    images.append(image)
    image_flipped = np.fliplr(image)
    images.append(image_flipped)
    
    measurement = float(line[3])
    measurement_flipped = -measurement
    measurements.append(measurement)
    measurements.append(measurement_flipped)
    
# Compile arrays of training data
X_train = np.array(images)
y_train = np.array(measurements)

# Create model
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# Normalize the images
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

# Crop the images to remove data from the image that can distract the network from predicting the lane
model.add(Cropping2D(cropping=((70, 24), (0, 0)))) #Output Image Shape: 3x66x320

# Add first convolutional layer
model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2))) #Input 3@66x200, Output 24@31x98
#model.add(MaxPooling2D())
#model.add(Dropout(0.25))

model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2))) #Input 24@31x98, Output 36@14x47
#model.add(MaxPooling2D())

model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2))) #Input 36@14x47, Output 48@5x22
#model.add(MaxPooling2D())

model.add(Conv2D(64, 3, 3, activation='relu')) #Input 48@5x22, Output 64@3x20
#model.add(MaxPooling2D())

# Add second convolutional layer
model.add(Conv2D(64, 3, 3, activation='relu')) #Input 64@3x20, Output 64@1x18
#model.add(MaxPooling2D())
#model.add(Dropout(0.25))
          
# Add fully connected layer
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
model.summary()
print("Run the command: python drive.py model.h5")
