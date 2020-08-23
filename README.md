# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Nvidia_network.png "Model Visualization"
[image2]: ./examples/model_summary.png "Model Summary"
[image3]: ./examples/MSE_plot.png "Model MSE loss plot"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model.
* fine-tune.py containing the script to fine tuning the model.
* drive.py for driving the car in autonomous mode.
* model.h5 containing a trained convolution neural network, first version.
* weight.h5 containing the first trained convolution neural network's weight, the weight     will be used for fine tuning the model.
* because the fine tuning process takes a long time, the code in fine-tune.py is copied to   keep_alive.py and keep the workspace awake until the training process finished.
* model_new.h5 containing the fine tuned convolution neural network.
* weight_new.h5 containing the fine tuned network's weight.
* Project_Writeup.md summarizing the results.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 run1
```
Using different model files to get different results. The model_new.h5 can drive autonomously around the full track.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

After the learning the lectures before starting this project, I chose to use Nvidia's end-to-end deep learning architecture for self-driving cars. The original modal contained 9 layers: 1 normalized layer; 5 convolutional layers; and 3 fully-connected layers. 

Here is a visualization of the architecture:

![model summary][image1]

I added another fully connected layer, Dense(1), because this network was solving a regression problem rather than a classification problem. The model summary is shown below.

![model summary][image2]

The model included RELU layers to introduce nonlinearity, and the data was normalized in the model using a Keras lambda layer, and cropped by Cropping2D layer. So, the original images were normalized first and had a input shape of 160 x 320. However, the top-half and bottom part of the 160 x 320 images didn't contain useful information. So, I cropped 70 rows pixels from the top of the image and 25 rows pixels from the bottom of the image. This made the input image with a size of 65 x 320.

#### 2. Attempts to reduce overfitting in the model

The model didn't contain dropout layers. Similar to the lecture, to ensure the model was not overfitting, I trained the model for 7 epochs first, and noticed the model's loss start to increasing after the fifth epoch. So, the number of epoch was recuded to 5 to prevent overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning and training data selection

The model used an adam optimizer, so the learning rate was not tuned manually.

For the training data, I used the project provided data. Because it contained a good amount of data samples and had multiple camera angles. At first, I tried to record my own data. But after drove and recorded for 2 laps, I still didn't collect enough data samples, and it was hard to keep the vehicle in the lane center. This made the collected data quality low. So, I chose to use the provided data.

#### 4. Appropriate training data

At first, I didn't use the generator to feed in the data. But without the generator, loading all images consumed lots of memory. So, I added a generator to feed the data. 

I splitted the dataset into 80% and 20%. 80% of the dataset was used for training the model. And 20% of the dataset was used for validation. The batch size was set to 128 and number of epoch was set to 5.

I only used the center camera images and steering angle to train the model. The model trained pretty fast, but after successfully passing 2 turns, the car stucked at the road curb. This situation illustrated that the trained model was not good at recovering once the car went out of track.

So, I made a fine tuning model to solve this issue. I saved the weight from the first model and used the saved weight for the fine tuning process. I used the left camera images and right camera images together, and used a correction factor of +0.2 and -0.2 for left steering angle and right steering angle accordingly. Further, from the lecture, more training data could be obtained by driving the vehicle in reverse direction. To accomplish this, I fliped the left camera image and right camera image to create more training samples. Also -1 times left&right steering angles to accommodate this change.

After adding more training samples to fine-tune the model, the training time became a lot longer. Previously, training for 1 epoch, it only took around 10 minutes. At fine-tuning stage, the training time for 1 epoch increased to around 1.5 hours.

After the fine tuning was finished, the model was able to drive through the full track autonomously, using the model_new.h5 and drive.py provided. The number of epoch equaled to 5 was proved to be reasonable, because the MSE loss continued decrasing and the fifth epoch had the lowest loss for trianing set. The model MSE loss plot is shown below.

![MSE Plot][image3]

