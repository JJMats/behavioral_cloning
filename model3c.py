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

#Decreasing this value helps improve vehicle stability at higher speeds
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
    #measurements.append(measurement + (measurement * corr_fact))
    #measurements.append(measurement - (measurement * corr_fact))

# Augment additional data to the training data to remove left turn bias
aug_images = []
aug_measurements = []

for img, meas in zip(images, measurements):
    aug_images.append(img)
    img_flipped = np.fliplr(img)
    aug_images.append(img_flipped)
    
    aug_measurements.append(meas)
    aug_measurements.append(-meas)

# Compile arrays of training data
X_train = np.array(aug_images)
y_train = np.array(aug_measurements)

# Create model
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, LeakyReLU
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

leaky_alpha = 0.1
keep_prob = 0.25

model = Sequential()

# Normalize the images
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
#model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))

# Crop the images to remove data from the image that can distract the network from predicting the lane
model.add(Cropping2D(cropping=((70, 24), (0, 0)))) #Output Image Shape: (66, 320, 3)

# First convolutional layer
model.add(Conv2D(24, 5, 5, subsample=(2, 2))) # Input: (66, 320, 3), Output: (31, 158, 24)
model.add(LeakyReLU(alpha=leaky_alpha))

# Second convolutional layer
model.add(Conv2D(36, 5, 5, subsample=(2, 2))) # Input: (31, 158, 24), Output: (14, 77, 36)
model.add(LeakyReLU(alpha=leaky_alpha))

# Third convolutional layer
model.add(Conv2D(48, 5, 5, subsample=(2, 2))) # Input: (14, 77, 36), Output: (5, 37, 48)
model.add(LeakyReLU(alpha=leaky_alpha))

# Fourth convolutional layer
model.add(Conv2D(64, 3, 3)) # Input: (5, 37, 48), Output: (3, 35, 64)
model.add(LeakyReLU(alpha=leaky_alpha))
#model.add(MaxPooling2D())

# Fifth convolutional layer
model.add(Conv2D(64, 3, 3)) # Input: (3, 35, 64), Output: (1, 33, 64)
model.add(LeakyReLU(alpha=leaky_alpha))

# Use dropout layer to reduce overfitting
model.add(Dropout(keep_prob))
          
# Add fully connected layers
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

model.save('model.h5')
model.summary()
print("Run the command: python drive.py model.h5")
