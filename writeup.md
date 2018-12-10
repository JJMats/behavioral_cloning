# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[imgModelVis]: ./output_images/misc/cnn-architecture.png "Model Visualization"
[imgRawLeft]: ./output_images/raw/raw_image1.png "Raw Image - Left Camera"
[imgRawCenter]: ./output_images/raw/raw_image0.png "Raw Image - Center Camera"
[imgRawRight]: ./output_images/raw/raw_image2.png "Raw Image - Right Camera"
[imgCropLeft]: ./output_images/cropped/cropped_image1.png "Cropped Image - Left Camera"
[imgCropCenter]: ./output_images/cropped/cropped_image0.png "Cropped Image - Center Camera"
[imgCropRight]: ./output_images/cropped/cropped_image2.png "Cropped Image - Right Camera"
[imgFlipLeft]: ./output_images/flipped/flipped_image1.png "Flipped Image - Left Camera"
[imgFlipCenter]: ./output_images/flipped/flipped_image0.png "Flipped Image - Center Camera"
[imgFlipRight]: ./output_images/flipped/flipped_image2.png "Flipped Image - Right Camera"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The chosen model consists of a convolutional neural network that is based upon the successful end-to-end deep learning model developed by NVIDIA. The structure of the network utilizes varying filter sizes, five convolutional feature layers, four fully connected layers, and multiple activation functions to reduce overfitting and improve the model's ability to generalize.


#### 2. Attempts to reduce overfitting in the model

Multiple approaches were used to reduce overfitting of the model. After numerous attempts, it was found that a single dropout layer in conjunction with the LeakyReLU activation functions provided a satisfactory reduction in neuron links to provide adequate performance of the final model. The model contains a dropout layer between the final convolutional layer and the flattening layer (model.py, line 112) to reduce overfitting of the model. 

For model validation, a data set was split from the training data before the model was trained to quantify the fitment of the model. An 80/20% split was chosen (training/validation) (model.py, line 23). After the model was trained, it was tested by utilizing the self-driving car simulator in autonomous mode to provide the test data for final validation of model performance. The performance of the model was quantified by ensuring that the vehicle could remain on the driving surface of the track while driving a lap around each course.


#### 3. Model parameter tuning

The hyperparameters utilized to tune the model include:

- Batch size
  - A generator was used to reduce the quantity of data loaded into memory throughout the training process. The final value chosen was 16 samples per batch. Smaller batch sizes improved training speed of the model considerably, but larger batch sizes improved training accuracy.
- Dropout keep probability
  - This value was chosen to reduce overfitting of the model. The range of values tested were 0.25 - 0.5, and the final selected value was 0.5. These affected how well the vehicle would follow the road and keep centered in the lane. Reducing this value tended to allow the vehicle to track closer to the lane edges when in turns, while increasing the value assisted in successful center finding.
- Epochs
  - The epoch count utilized was dependent upon the training data set size and the batch size. Increasing the batch size tended to slow model training down, but improved the accuracy. Therefore, this value could be reduced when the batch size was increased (inversely proportional).
- LeakyReLU alpha (leaky_alpha)
  - The alpha value chosen for the LeakyReLU activation function was tuned by visualizing the vehicle driving autonomously around the test tracks. Values tested ranged from 0.1 to 0.3, but reducing the value improved lane keeping and center finding.
- Learning rate
  - The model used an Adam optimizer, so the learning rate was set by default at 0.001 (model.py, line 121).
- Steering correction factor
  - All three of the camera views obtained from the simulator were utilized for training and validation of the model. A steering correction factor was applied to the steering angle measurement (pre-training) to improve vehicle straight-line stability and to adjust the gain of the steering angle influence on the model provided by the side camera images. Values tested ranged from 0.1 to 0.3, and the median value of 0.2 was settled upon.


#### 4. Appropriate training data

Model training data was generated by utilizing the training mode of the Udacity Self-Driving Car Simulator. It was collected while manually driving the vehicle through multiple laps on each of the tracks. This data collection featured a combination of center lane driving, recovery from the left and right sides of the road back to the lane center, and laps obtained in both directions of the track.

For details about how I created the training data, see the next section. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to generate a neural network that was capable of driving a vehicle based upon behavioral cloning, instead of a robotics approach.

To begin, my initial step was to use a convolutional neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because it has been utilized successfully in other autonomous vehicle endeavors, including a remote controlled car project.

To quantify the model's ability to generalize, I split the training data (images and steering angles) into training and validation data sets. I chose a 80/20% split of the data, and used the larger portion for training the model. Upon initial validation, it was found that the MSE (mean squared error) calculated on the training set was low, but the MSE of the validation data was slightly more. This implies that the model was slightly overfitting the data, so it was worthwhile to make attempts to reduce it.

To reduce the model's tendency to overfit the data, I selected different activation layers (changed from ReLU to LeakyReLU) and added a dropout layer. 

After the model was trained and validated, it was then necessary to run the simulator to visualize the performance of the network on images input from a vehicle driving autonomously around the track. During the first test on the "Lake" track, the vehicle found the lane center fairly well on straight sections of the track, but did tend to drive off of the side of the road in turns, and in locations where the features at the side of the driving surface could be misjudged as a viable path.

To improve the behavior of the model from the testing results, I spent time tuning the hyperparameters, of which the LeakyReLU "alpha" and dropout "keep probability" tended to be the most effective. I was able to get the vehicle to drive successfully near the center of the lane on the "Lake" track with this tuning alone. To obtain a successful run on the "Jungle" track, it was necessary to gather additional training data from that track.

Upon conclusion of the process, the vehicle was able to drive autonomously on both of the simulator tracks without leaving the road. It was able to maintain it's position in the center of the driving surface fairly well during a large majority of the lap.


#### 2. Final Model Architecture

The final model architecture (model.py, code lines 78-134) begins with a normalization layer. This layer converts the 8-bit pixel values from each of the three color channels (RGB), and reduces them to a zero-centric value between -0.5 and +0.5. This helps to reduce the work the optimizer must do because all of the pixel values are within a narrowed range around a zero median value. This is performed by utilizing a Keras Lambda layer (code line 86). An attempt was made to increase the range of the normalized values to +/-1.0, but this reduced the stability of the model.

The next layer crops the images into smaller sizes. The images generated by the simulator are 120px X 320px, and they are reduced to 66px X 320 px by utilizing a Keras Cropping 2D layer (code line 89). This has multiple benefits, since it reduces the workload on the model, and removes extraneous objects from the field of view that either do not help the model, or negatively affect the decisions made. The top of each image was trimmed by 70 pixels to remove distant background features while improving the ability of the model to handle positive and negative slopes, and the bottom of the image was trimmed by 24 pixels to remove the hood of the car. An example of a raw simulator output image and a cropped image are as follows:

**** Images Here (Raw Image, Cropped Image) 

Following the cropping layer, five convolutional layers were implemented. These utilize the Keras Convolutional2D (Conv2D in Keras 2) layer, and are each followed by a Keras LeakyReLU activation function. Here is a visualization of the architecture:

![alt text][imgModelVis]
_ Original Source: NVIDIA CNN Neural Network Architecture. NVIDIA Developer, NVIDIA Developer Blog, 17 Aug. 2016,
	https://devblogs.nvidia.com/deep-learning-self-driving-cars _

The first three layers apply a convolution to the image with 5x5px filters, a 2x2px stride, and 'VALID' padding (code lines 92, 96, 100). Through this process, the input is transformed from the 66x320px 3-layer cropped image to a 31x158px sized 24-layer sample, and then to a 14x77px sized 36-layer sample.

The next two convolutional layers are applied with a 3x3px filter, a 1x1px stride, and 'VALID' padding (code lines 104, 108). These layers reduce the sample size further, while continuing to increase the depth. The sample is transformed from the 14x77px sized 36-layer version from the third layer into a 5x37px sized 48-layer version, and then finally into a 3x35px sized 64-layer sample.

In-between each of the convolutional layers, a LeakyReLU activation function is applied to assist in reducing overfitting of the model. The LeakyReLU function was chosen because...

The model then flattens the sample into a 1x33px sized 64-layer object and passes it through three fully-connected Keras Dense layers (code lines 116-119). These fully-connected layers reduce the object from 2112 neurons to 100, 50, and then 10, respectively. The final output is then a single neuron that produces the predicted steering angle to be utilized by the autonomous driving simulator. 

The model consists of 348,219 trainable parameters, and achieves a training MSE of 0.0243 with a validation MSE of 0.0894.


#### 3. Creation of the Training Set & Training Process

To train the model, it was necessary to capture ideal driving behavior. To begin, three laps were recorded on the "Lake" track using center lane driving. Another lap was captured travelling in the opposite direction of the first three to help minimize turning direction bias. An example image set captured of center lane driving is:

![alt text][imgRawLeft]
![alt text][imgRawCenter]
![alt text][imgRawRight]

In addition to the center lane driving data, samples were captured of the vehicle recovering from the left and right sides of the driving surface back to the center to help the network learn to steer back to the center of the driving surface if its position started to deviate towards an edge.

The samples gathered from the "Lake" track training were adequate to generate a network that performed well on that particular venue. The model did not perform well on the "Jungle" track because of the slopes, sharp turns, cliffs, and other background features that were very different from the prior test scenario. It was necessary to repeat the sample gathering on the "Jungle" track. Three laps in one direction were recorded, followed by two laps in the opposing direction, and this was finalized with extra samples of driving through sections of the track that were particularly difficult drive through manually.

To further augment the data set, all of the images gathered from each of the three cameras were flipped left-to-right, and the steering angle measurements were negated. This doubled the sample size in an effort to increase training data while reducing left/right turning bias. An example of an image set that has been flipped is:

![alt text][imgFlipLeft]
![alt text][imgFlipCenter]
![alt text][imgFlipRight]

Upon completion of the training data collection process, the resulting training image count was 105,420 images (17,570 samples x 3 camera views x 2 flipped image augmentation). Further preprocessing of the data included cropping and normalizing the images. An example of an image set that has been cropped is:

![alt text][imgCropLeft]
![alt text][imgCropCenter]
![alt text][imgCropRight]

The data set was then shuffled, and 20% of it was split off into a validation data set. Finally, the model was trained utilizing the training data set, and the fitment was quantified with the validation set. It was determined that four epochs were adequate to train the model with a slight overfitment as evidenced by a decreasing MSE on the training data while the MSE on the validation data slightly increased. This value was also balanced against training time of the model, as batch selection size from this large data set affected training duration and accuracy dramatically.

An Adam optimizer was selected because it is computationally efficient, and can provide satisfactory results rapidly. It also decays the learning rate as training progresses which can improve overall accuracy of the model as opposed to a fast, fixed learning rate.