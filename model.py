import csv
import cv2
import numpy as np
from scipy import ndimage

lines = []

# Directory to training data
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
    for i in range(3):        
        source_path = line[i]
        tokens = source_path.split('/')
        filename = tokens[-1]
        local_path = directory + 'IMG/' + filename
        
        image = ndimage.imread(local_path)
        images.append(image)
    
    measurement = float(line[3])    
    measurements.append(measurement)
    measurements.append(measurement + corr_fact)
    measurements.append(measurement - corr_fact)
    
# Compile arrays of training data
X_train = np.array(images)
y_train = np.array(measurements)

# Create model
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, LeakyReLU
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

leaky_alpha = 0.1
keep_prob = 0.2

model = Sequential()

# Normalize the images
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

# Crop the images to remove data from the image that can distract the network from predicting the lane
model.add(Cropping2D(cropping=((70, 24), (0, 0)))) #Output Image Shape: 3x66x320

# Add convolutional layers
model.add(Conv2D(24, 5, 5, subsample=(2, 2))) #Input 3@66x320, Output 24@31x158
model.add(LeakyReLU(alpha=leaky_alpha))

model.add(Conv2D(36, 5, 5, subsample=(2, 2))) #Input 24@31x158, Output 36@14x77
model.add(LeakyReLU(alpha=leaky_alpha))

model.add(Conv2D(48, 5, 5, subsample=(2, 2))) #Input 36@14x77, Output 48@5x37
model.add(LeakyReLU(alpha=leaky_alpha))

model.add(Conv2D(64, 3, 3))  #Input 48@5x37, Output 64@3x35
model.add(LeakyReLU(alpha=leaky_alpha))

model.add(Conv2D(64, 3, 3)) #Input 64@3x35, Output 64@1x33
model.add(LeakyReLU(alpha=leaky_alpha))

model.add(Dropout(keep_prob))

# Add fully connected layer
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')
model.summary()
print("Run the command: python drive.py model.h5")
