#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[histo]: ./images/histo.png "Redistrubtion of the training set"
[step1]: ./images/step1.png "Step1"
[step2]: ./images/step2.png "Step2"
[step3]: ./images/step3.png "Step3"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* models.py containing the script to create the model
* train.py to feed the model and train it
* import_data.py taking care of data reading, redistribution & generator
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The train.py & models.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

After first trying out the LeNet model, I had underfit problems leading to poor performance on the track. That was back when I didn't have all the preprocessing steps detailed later, which means that this model might work as well now that I have proper data conditionning and it has the advantage of being lighter that the Nvidia one so this is something to try again when thinking about realtime applications.

My model consists of a convolution neural network based on the nvidia architecture (models.py lines 53-70). 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 12). The input data is cropped to include only the part of the image which contains relevant information to decide which direction to steer to.

Cropped image

![Cropped sample image][step1]

I included as well a preprocessing step which is not directly done in the model itself but in data reading as I couldn't implement it yet in Keras or Tensorflow: converting the image from RGB to HSV and then keeping only the Hue channel. The idea is the following: the Hue channel contains the chroma, better seperated from the lumina information, which should make this colorspace more robust to lighting variations contrary to RGB. The Hue channel shall contain enough information to drive while reducing the amount of inputs so that the DNN can focus on stricly vital information to decide (less noise).

HSV image

![HSV sample image][step2]

Hue channel only

![Hue channel only][step3]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The import_data.py file contains an important "redistribute" function which looks over the histogram of all angles used for training and excluses randomly samples from each bin which has a count over a certain cap threshold parameter.
This produces a flatter distribution of the training samples so that the model is trained more equally on all cases.

![Sample Redistribution][histo]
The model was trained and validated on different data sets to ensure that the model was not overfitting (train.py line 33-37). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I focused exclusively on driving at the center of the road and compensated using the side cameras and applying a correction offset. I used the provided data set along with 2 other recordings per track: one forward and one backward to provide enough data to train my model.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to gradually complexify the model to make sure it could overfit the data. I used a CNN base as this architecture demonstrated good performance on previous image classification tasks in the convolutionnal layer and the Fully connected layers shall be enough to use high level features such as road delimiters to steer the car properly.

Given the first data sets I had recorded myself, my modified LeNet architecture from the previous project could not achieve this so I went up to the Nvidia model which was enough, even probably too much for the task of this project as it quickly shows overfit.

From that basis, after applying the proposed preprocessing methods from the introduction video, I looked at options to preprocess the image to give the network only what it absolutely needed to decide where to drive. One of the key improvement I noticed was when working with only the H component of the image after converting from RGB to HSV (see section 2. for the idea behind). This although speeded up training as my input was now 1dimension instead of 3 in depth.

To combat the overfitting, I focused on recording more accurate data and I modified the distribution of angle samples to make it much more flat so that the model would train on a well distributed set of examples.
I simply used early termination of the training (only 5epochs) to prevent overfitting but due to the lack of time, I couldn't do the following which showed very good improvements in the previous project:
- use a high keep prob in CN layers (0.95) as due to weight sharing, they are not too likely to overfit
- use a high keep prob in FC layers (0.5) as they are much more likely to cause overfit based on the still generic features of the CN layers output
- if still not enough, adding some L2 normalization factor

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road on track1. It can even drive correcly at high speeds than 9, for example 20 will do but it will start looking like a drunk driver. 15 was still producing smooth results.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted directly of the Nvidia model. The major difficulty in this project to me was to correctly choose and preprocess the data rather than choosing the correct network architecture as a MSE with a low distance between training set and test set didn't mean the car would drive ok on both tracks.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

See Appropriate training data section for the kind of data I recorded to train my model on both tracks.

To augment the data sat, I also flipped images and angles thinking that this would prevent overfit by showing more cases in training just as if it was a new track. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I also recorded both tracks in "reverse" to give again more variety to my model and flipped this images too.
The side camera images were used and flipped as well.

After the collection process, I had X number of data points. As explained above, this data set was adjusted to be better distributed by looking at the overpopulated bins of a histogram done over the whole data set.

After adjusting the dataset distribution, this gave me a final number of Z samples.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the loss plot below (early termination to avoid overfitting). I used an adam optimizer so that manually training the learning rate wasn't necessary.
