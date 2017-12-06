**Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[original]: ./examples/original.png "Original Image"
[original_BGR]: ./examples/original_BGR.png "Original Image BGR"
[original_cropped]: ./examples/original_cropped.png "Cropped Image"
[original_cropped_resize]: ./examples/original_cropped_resized.png "Cropped Image resized"
[center]: ./examples/center.jpg "Center driving second track"
[transformations]: ./examples/transformations.png "Transformations"


### Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model, the model.py can be run as explained in the following example:
```sh
python model.py "path/to/driving_log.csv" "path/to/images/directory" "model.h5" batch_size learning_rate epochs
```

* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the first and second
 track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline 
I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

My model is a convolution neural network that consists of 3 convolution layers with 5x5 filters
 and 2x2 max pooling between them, then 2 convolution layers with 3x3 filters and one 2x2 max pooling layer between them.
 The depth of the convolution layers ranges from 24 to 64 (model.py lines 134-175) 

The model includes ReLU layers to introduce non-linearity, and the data is normalized in the model using a Keras lambda layer (code line 136). 
The architecture of my model is as follows:

| Layer (type)                    | Output Shape         | Param #                        
|:-------------------------------:|:--------------------:|:----------:
| Normalization (Lambda)          |  (64, 64, 3)    | 0                     
| (Convolution2D 5x5)             |  (32, 32, 24)   | 1824                  
| (ReLU Activation)               |  (32, 32, 24)   | 0                    
| (MaxPooling2D 2x2)              |  (31, 31, 24)   | 0                  
| (Convolution2D 5x5)             |  (16, 16, 36)   | 21636              
| (ReLU Activation)               |  (16, 16, 36)   | 0                  
| (MaxPooling2D 2x2)              |  (15, 15, 36)   | 0                 
| (Convolution2D 5x5)             |  (8, 8, 48)     | 43248                
| (ReLU Activation)               |  (8, 8, 48)     | 0                    
| (MaxPooling2D 2x2)              |  (7, 7, 48)     | 0                    
| (Convolution2D 3x3)             |  (7, 7, 64)     | 27712                  
| (ReLU Activation)               |  (7, 7, 64)     | 0                      
| (MaxPooling2D 2x2)              |  (6, 6, 64)     | 0                         
| (Convolution2D 3x3)             |  (6, 6, 64)     | 36928                      
| (ReLU Activation)               |  (6, 6, 64)     | 0                                               
| (Flatten)                       |  (2304)         | 0                        
| (Dense)                         |  (1164)         | 2683020                      
| (ReLU Activation)               |  (1164)         | 0                              
| (Dense)                         |  (100)          | 116500                    
| (ReLU Activation)               |  (100)          | 0                             
| (Dense)                         |  (50)           | 5050                       
| (ReLU Activation)               |  (50)           | 0                           
| (Dense)                         |  (10)           | 510                        
| (ReLU Activation)               |  (10)           | 0                             
| (Dense)                         |  (1)            | 11                      

My model is based on Nvidia's "End to End Learning for Self-Driving Cars" paper.
I have tried to implement the model verbatim, but for some reason did not yield good control of the
vehicle, the change between the angles was smooth but when it came to sharp turns it wasn't good
 enough. This is also was true when I used ELU activation layer rather than ReLU, smooth angles but not
 good enough for sharp turns.

#### 2. Attempts to reduce overfitting in the model

I have tried to use dropout layers in order to reduce over-fitting but my model wasn't over-fitting the data 
 since my generator was generating random images each time by apply random transformations. 
 The same generator was used for training and testing the difference was the data. 

The model was trained and validated on different data sets to ensure that the model was not over-fitting (code line 119). 
I used ```train_test_split()``` function with test ratio 20% of the training samples available in the data set.
A
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track one and track two.

#### 3. Model parameter tuning

The model used an adam optimizer, with learning rate of 1e-4 which yielded better validation accuracy
than the default value of 0.001 (model.py line 177).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road specially in the first track
 however, I used the second track to balance the ratio of the left and right steering angles to the straight driving.
 
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a known model that works for this kind of problem
and start to tweak it as necessary.

My first step was to use a convolution neural network model similar to the one presented in the Nvidia paper. 
I thought this model might be appropriate because it was used to solve a similar problem however, when I implemented this model I used ELU
as activation layer with no max pooling. It seemed to work with some other students who posted their work online. As I explained earlier, 
this model yielded smooth turns but were not sharp enough for some of the sharp turns in the first and most of the second track. I guess it
was lack of training data. Then I implemented the Comma.ai model, which is available online, same as the initial Nvidia model, didn't work for me.
Then I implemented the model that I have right now, also didn't work. At that point I decided to perform data analysis which is done
mostly in the IPython notebook that is included with this project. I found that data is not balanced and that was the reason of the 
bias towards left and straight driving. To solve this problem, I have implemented two steps, first, I used the data from the second track,
the second track does not have that bias found in the first track, the second step, was in the code where I draw a random number between 
0 and 2 inclusive to train the model with either of the right, center, or left image. The selected image was flipped randomly as well.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
Although the models that I have implemented achieved low mean squared error on the validation set, when testing in the simulator, the vehicle went off track.  
I kept trying and trying to tweak the parameters, adding and removing layers and different types of activation layers, nothing worked,
it took me several days to realize that the data set that I was using (which was based on the Udacity's sample data) is not enough to train
my model properly. So I started from scratch and recorded my own data on both tracks with some additional recovery turns from sharp curves. 

I never faced over-fitting problem in this project, in part because the data set wasn't that big and the ```preprocess_img``` function generates
random variant of same image every time it is called.

The biggest lesson here was enough balanced data is very important for training any model properly.


The final step was to run the simulator to see how well the car was driving around track one. 
There were a few spots where the vehicle fell off the track so I recorded recovery sessions to improve the driving behavior in these cases, 
I also tweaked the correction angle that is added or subtracted from left and right camera images.

At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road.

#### 2. Final Model Architecture

The final model architecture:

| Layer (type)                    | Output Shape         | Param #                        
|:-------------------------------:|:--------------------:|:----------:
| Normalization (Lambda)          |  (64, 64, 3)    | 0                     
| (Convolution2D 5x5)             |  (32, 32, 24)   | 1824                  
| (ReLU Activation)               |  (32, 32, 24)   | 0                    
| (MaxPooling2D 2x2)              |  (31, 31, 24)   | 0                  
| (Convolution2D 5x5)             |  (16, 16, 36)   | 21636              
| (ReLU Activation)               |  (16, 16, 36)   | 0                  
| (MaxPooling2D 2x2)              |  (15, 15, 36)   | 0                 
| (Convolution2D 5x5)             |  (8, 8, 48)     | 43248                
| (ReLU Activation)               |  (8, 8, 48)     | 0                    
| (MaxPooling2D 2x2)              |  (7, 7, 48)     | 0                    
| (Convolution2D 3x3)             |  (7, 7, 64)     | 27712                  
| (ReLU Activation)               |  (7, 7, 64)     | 0                      
| (MaxPooling2D 2x2)              |  (6, 6, 64)     | 0                         
| (Convolution2D 3x3)             |  (6, 6, 64)     | 36928                      
| (ReLU Activation)               |  (6, 6, 64)     | 0                                               
| (Flatten)                       |  (2304)         | 0                        
| (Dense)                         |  (1164)         | 2683020                      
| (ReLU Activation)               |  (1164)         | 0                              
| (Dense)                         |  (100)          | 116500                    
| (ReLU Activation)               |  (100)          | 0                             
| (Dense)                         |  (50)           | 5050                       
| (ReLU Activation)               |  (50)           | 0                           
| (Dense)                         |  (10)           | 510                        
| (ReLU Activation)               |  (10)           | 0                             
| (Dense)                         |  (1)            | 11                      


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 2 or 3 laps on track two using center lane driving. 
Here is an example image of center lane driving in the second track:

![alt text][center]

Then I recorded 1 or 2 laps from track one.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to
recover from sharp curves. In particular the sharp curve right after the bridge in the first track, and two other curves that come 
after it.

I have applied several transformations on the image as suggested by several articles that I read online and by what I have implemented
before in the traffic sign recognition project. I have implemented the following transformations:
- Changed the color map from BGR to RGB, here is an example:

![alt_text][original_BGR]
![alt text][original]

- Cropping the image 60 pixels from top and 25 pixels from bottom. 
So that we don't distract the model with the sky and the hood of the car. Here is an example original and cropped version:

![alt text][original]
![alt_text][original_cropped]
- Re-sized the image to 64x64 pixels, here is an example:

![alt_text][original_cropped_resize]

- Random brightness
- Random translation (shift) in horizontal axis between -25 and 25 pixels, for each pixel shift I modified the steering angle by 0.004 value. This was suggested by 
- Random lightness
- Random saturation
- Random gaussian blur
- Flipping of the image and the corresponding steering angle 

Here is an example of applying the remaining transformations on an image with an original angle of -0.15:

![alt_text][transformations]

The flipping of images and angles was applied because I thought that this would reduce bias towards left angle driving.

After the collection process, I had about 20,937 number of data samples. Each data sample contains right, center, and left camera image.


I finally randomly shuffled the data set and put 20% (4188) of the data into a validation set and the remaining data samples were used
for training the model.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 10 as evidenced by validation loss wasn't improving any more.
I used an adam optimizer with initial learning rate of 1e-4.
