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
[histo]: ./images/histo.png "Redistrubtion of the training set"
[original]: ./images/original.png "Step1"
[cropped]: ./images/cropped.png "Step2"
[H_chan]: ./images/H_chan.png "H"
[S_chan]: ./images/S_chan.png "S"
[V_chan]: ./images/V_chan.png "V"
[loss]: ./images/loss_curve.png "loss"

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

My model consists of a convolution neural network based on the nvidia architecture (models.py lines 53-70) but slightly simplified: I removed the 2 last Conv. layers to make it less sensitive to overfitting. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 12). The input data is cropped to include only the part of the image which contains relevant information to decide which direction to steer to.


![Original image][original]
![Cropped image][cropped]

I included as well a preprocessing step which is not directly done in the model itself but in data reading as I couldn't implement it yet in Keras or Tensorflow: converting the image from RGB to HSV and then keeping only the Hue channel. The idea is the following: the Hue channel contains the chroma, the S channel contains the saturation and the V channel contains the value. From the example below, it's pretty clear that the V channel is very close to what we see individually in a R G or B channel but the H and S have a very different view of the road and better seperates the lumina information from the color information, which should make this colorspace more robust to lighting variations contrary to RGB.

![H channel][H_chan]
![S channel][S_chan]
![V channel][V_chan]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. It has as well less Conv. layers to reduce complexity.

The import_data.py file contains an important "redistribute" function which looks over the histogram of all angles used for training and excluses randomly samples from each bin which has a count over a certain cap threshold parameter.
This produces a flatter distribution of the training samples so that the model is trained more equally on all cases.

![Sample Redistribution][histo]

The model was trained and validated on different data sets to ensure that the model was not overfitting (train.py line 33-37). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I focused exclusively on driving at the center of the road and compensated using the side cameras and applying a correction offset. I used the provided data set along with 2 other recordings per track: one forward and one backward to provide enough data to train my model.
As for track 2, I had one lap forward and one lap backward as well as another set of data which captures sharp turns towards the end of the track to give the model a pinch more of high steering wheel values.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to gradually complexify the model to make sure it could overfit the data. I used a CNN base as this architecture demonstrated good performance on previous image classification tasks in the convolutionnal layer and the Fully connected layers shall be enough to use high level features such as road delimiters to steer the car properly.

Given the first data sets I had recorded myself, my modified LeNet architecture from the previous project could not achieve this so I went up to the Nvidia model which was enough for track1 but cause me quite some trouble on track2: it was too powerful to tackle this problem without having hours of driving: I had to simplify it else it would consitently overfit in my case.

From that basis, after applying the proposed preprocessing methods from the introduction video, I looked at options to preprocess the image to give the network only what it absolutely needed to decide where to drive. One of the key improvement I noticed was when converting from RGB to HSV (see section 2. for the idea behind). I even think that keeping only the H and S channel would be enough as the V channel was too close to the RGB image, something to be tried.

To combat the overfitting, I focused on recording more accurate data and I modified the distribution of angle samples to make it much more flat so that the model would train on a well distributed set of examples.
I added as well dropout on the FC layers (0.5) as they are much more likely to cause overfit based on the still generic features of the CN layers output.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road on track1. It can even drive correcly at high speeds than 9, for example 20 will do but it will start looking like a drunk driver. 15 was still producing smooth results.

The real work came from track2: my vehicle would drive off the road at sharp turns towards the end of the track. I then recorded more data just focused on these sharp turns but still had problems later on the track and recording more data on these would then create more problems ! I manage to solve the neverending recording by coming reducing the complexity of the Nvidia model and removing the 2 last Conv. layers. 

After that, I was able to drive properly on both tracks only adding sharp turns recording to have more samples of high steering angles.

#### 2. Final Model Architecture

The final model architecture (model.py lines 73-93) consisted of a simplfied version of the Nvidia model as stated above. The major difficulty in this project to me was to correctly choose and preprocess the data and have a model simple enough to avoid having hours of data recording as an MSE with a low distance between training set and test set didn't mean the car would drive ok on both tracks.

|Layer (type)           |          Output Shape     |     Param #   |  Connected to |                     
|:---------------------:|:--------------------------:|:--------------:|:-----:| 
| lambda_1 (Lambda)        |        (None, 160, 320, 3)|   0        |   lambda_input_1[0][0]       |      
|cropping2d_1 (Cropping2D)    |    (None, 90, 320, 3)  |  0 |      |    lambda_1[0][0]           |      
|convolution2d_1 (Convolution2D) | (None, 43, 158, 24) |  1824      |  cropping2d_1[0][0]          |     
|convolution2d_2 (Convolution2D) | (None, 20, 77, 36)  |  21636     |  convolution2d_1[0][0]       |     
|convolution2d_3 (Convolution2D) | (None, 8, 37, 48)   |  43248     |  convolution2d_2[0][0]       |    
|flatten_1 (Flatten)             | (None, 14208)       |  0        |   convolution2d_3[0][0]       |     
|dense_1 (Dense)                |  (None, 100)         |  1420900   |  flatten_1[0][0]             |     
|dropout_1 (Dropout)            |  (None, 100)         |  0         |  dense_1[0][0]               |     
|dense_2 (Dense)                |  (None, 50)         |   5050     |   dropout_1[0][0]              |  
|dropout_2 (Dropout)            |  (None, 50)          |  0       |    dense_2[0][0]                |    
|dense_3 (Dense)                 | (None, 10)         |   510     |    dropout_2[0][0]               |   
|dense_4 (Dense)                |  (None, 1)          |   11      |    dense_3[0][0]                 |   

Total params: 1,493,179
Trainable params: 1,493,179


#### 3. Creation of the Training Set & Training Process

See Appropriate training data section for the kind of data I recorded to train my model on both tracks.

To augment the data sat, I also flipped images and angles thinking that this would prevent overfit by showing more cases in training just as if it was a new track.

I also recorded both tracks in "reverse" to give again more variety to my model and flipped this images too.
The side camera images were used and flipped as well.

After the collection process, I had 68595 number of data points. As explained above, this data set was adjusted to be better distributed by looking at the overpopulated bins of a histogram done over the whole data set. After redistrubtion, that gave me 44412 and after a split of 20% for validation and augmentation I ended up with a final number of 71058 samples.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the loss plot below: the simplified along with dropout needed more epochs to converge to some good performances. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![Loss curves][loss]

#### 4. List of things to do with more time
- Reduce the model size and complexity: less parameters might be needed to tackle both these tracks even though I used fantastic details mode in the simulator to add up more realism
- Reduce the image size: a lower resolution could be enough
- Use only H and S channels as they have more relevant information than V or RGB
- Record data to keep vehicle on the right side of the road intead of driving in the middle on track 2 
- I could spend hours more on this problem and try my model on GTA V but my wife is already pissed so I shall stop here....
