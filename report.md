# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[dave-2]: report/nvidia_cnn_architecture.png "NVidia DAVE-2"
[training-data-1]: report/training-data-1.png "training-data-1"
[training-data-2]: report/training-data-2.png "training-data-2"
[training-data-3]: report/training-data-3.png "training-data-3"
[training-data-4]: report/training-data-4.png "training-data-4"
[training-data-5]: report/training-data-5.png "training-data-5"
[training-data-6]: report/training-data-6.png "training-data-6"
[training-data-7]: report/training-data-7.png "training-data-7"
[training-data-8]: report/training-data-8.png "training-data-8"
[training-data-9]: report/training-data-9.png "training-data-9"
[training-data-10]: report/training-data-10.png "training-data-10"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb containing the Jupyter notebook to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 video of simulator running in autonomous mode

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

Note that I changed the following from the standard:
* [CarND-Term1-Starter-Kit](https://github.com/boardthatpowder/CarND-Term1-Starter-Kit.git) updated Python, TensorFlow and Keras versions to match the AWS Deep Learning AMI
* `driver.py` updated to convert the RGB image to YUV colorspace to match my model

#### 3. Submission code is usable and readable

The `model.ipynb` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the _DAVE-2 NVidia convolution neural network_ as described in the [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf) paper.  In addition I adding ELU activation layers after each convolution and fully connected layer, along with adding a Dropout layer after each fully connected layer.

Refer to the `NVidia V2` section in `P3.ipynb`.

I added cropping and resizing to my model too to make it easier for the provided `drive.py` autonomous simulator to consume the model.

#### 2. Attempts to reduce overfitting in the model

As described above, the model contains Dropout layers occuring after the 3 fully-connected layers, each configured to drop 50%.

The model was trained and validated on different data sets to ensure that the model was not overfitting. When training the model I configured an EarlyStopping callback to stop the training early after 5 consecutive epochs with no validation loss improvement.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

I used a `rmsprop` optimizer with the default learning rate of `0.001`.  I also experimented with an `adam` optimizer, but achieved a better result with the `rmsprop` optimizer.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of anto-clockwise center lane driving, clockwise center lane driving, recovering from the left and right sides of the road, and collected extra driving data focused on driving across bridges and driving around corners with no barriers (the dirt tracks).

For each center image, I also used the corresponding left and right camera images, subtracting/altering 0.2 to the center angle to compensate.  I also flipped each of the images (along with angles) to augment the data.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with a standard LeNet model, but then moved to the NVidia DAVE-2 model as encouraged by the course study material.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I introduced Dropout layers, and added an EarlyStopping callback to stop my training when I had encountered 5 consecutive epochs with no improvement in the validation loss.  

During prediction, I was not happy with the intial quality as the car was driving off the track in multiple places therefore augmented the data as described in the _Appropriate training data_ section.  I experimented with adjusting angles for the left and right camera images from+/0.1 to +/-0.3, but settled on +/-0.2 as that achieved the best results.

To match the input of the NVidia DAVE-2 model exactly, I preprocessed the training data to convert from the expected RGB color space to YUV.  Then as part of the model itself I cropped and resized the images to 66x200.

Finally I experimented with the activation layers, originally adding ReLU layers after each Convoutional layer, but then finally adding ELU layers to all layers except the output later.

After each change I made to y model, I ran the simulator to see how well the car was driving around track one. The hardest problem I had to solve was keeping the car in the track when it encountered sections of the road with no side barriers (the dirt tracks).  I found that the conversion of images from RGB to YUV colorspace along with increased training data for these specific sections of road helped solve this problem.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

I based my model on the NVidia DAVE-2 model:

![dave-2][dave-2]


A summary of my model implementation is as follows:

```sh
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
norm (Lambda)                (None, 160, 320, 3)       0         
_________________________________________________________________
crop (Cropping2D)            (None, 80, 320, 3)        0         
_________________________________________________________________
resize (Lambda)              (None, 66, 200, 3)        0         
_________________________________________________________________
conv1 (Conv2D)               (None, 31, 98, 24)        1824      
_________________________________________________________________
elu1 (ELU)                   (None, 31, 98, 24)        0         
_________________________________________________________________
conv2 (Conv2D)               (None, 14, 47, 36)        21636     
_________________________________________________________________
elu2 (ELU)                   (None, 14, 47, 36)        0         
_________________________________________________________________
conv3 (Conv2D)               (None, 5, 22, 48)         43248     
_________________________________________________________________
elu3 (ELU)                   (None, 5, 22, 48)         0         
_________________________________________________________________
conv4 (Conv2D)               (None, 3, 20, 64)         27712     
_________________________________________________________________
elu4 (ELU)                   (None, 3, 20, 64)         0         
_________________________________________________________________
conv5 (Conv2D)               (None, 1, 18, 64)         36928     
_________________________________________________________________
elu5 (ELU)                   (None, 1, 18, 64)         0         
_________________________________________________________________
flatten (Flatten)            (None, 1152)              0         
_________________________________________________________________
dense1 (Dense)               (None, 100)               115300    
_________________________________________________________________
drop1 (Dropout)              (None, 100)               0         
_________________________________________________________________
elu6 (ELU)                   (None, 100)               0         
_________________________________________________________________
dense2 (Dense)               (None, 50)                5050      
_________________________________________________________________
drop2 (Dropout)              (None, 50)                0         
_________________________________________________________________
elu7 (ELU)                   (None, 50)                0         
_________________________________________________________________
dense3 (Dense)               (None, 10)                510       
_________________________________________________________________
drop3 (Dropout)              (None, 10)                0         
_________________________________________________________________
elu8 (ELU)                   (None, 10)                0         
_________________________________________________________________
output (Dense)               (None, 1)                 11        
=================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
_________________________________________________________________

```

I ran the model until the EarlyStopping callback considered it to be no longer improving (40 epochs) as follows:

```
Epoch 1/100
58/57 [==============================] - 61s 1s/step - loss: 0.0487 - val_loss: 0.0265

Epoch 00001: val_loss improved from inf to 0.02655, saving model to model-nvidia-v2-006.h5
Epoch 2/100
58/57 [==============================] - 52s 902ms/step - loss: 0.0316 - val_loss: 0.0236

Epoch 00002: val_loss improved from 0.02655 to 0.02360, saving model to model-nvidia-v2-006.h5
Epoch 3/100
58/57 [==============================] - 53s 913ms/step - loss: 0.0280 - val_loss: 0.0221

Epoch 00003: val_loss improved from 0.02360 to 0.02209, saving model to model-nvidia-v2-006.h5
Epoch 4/100
58/57 [==============================] - 53s 913ms/step - loss: 0.0260 - val_loss: 0.0224

Epoch 00004: val_loss did not improve from 0.02209
Epoch 5/100
58/57 [==============================] - 53s 911ms/step - loss: 0.0251 - val_loss: 0.0210

Epoch 00005: val_loss improved from 0.02209 to 0.02103, saving model to model-nvidia-v2-006.h5
Epoch 6/100
58/57 [==============================] - 54s 925ms/step - loss: 0.0238 - val_loss: 0.0203

Epoch 00006: val_loss improved from 0.02103 to 0.02026, saving model to model-nvidia-v2-006.h5
Epoch 7/100
58/57 [==============================] - 53s 912ms/step - loss: 0.0229 - val_loss: 0.0214

Epoch 00007: val_loss did not improve from 0.02026
Epoch 8/100
58/57 [==============================] - 54s 926ms/step - loss: 0.0224 - val_loss: 0.0201

Epoch 00008: val_loss improved from 0.02026 to 0.02005, saving model to model-nvidia-v2-006.h5
Epoch 9/100
58/57 [==============================] - 52s 900ms/step - loss: 0.0216 - val_loss: 0.0196

Epoch 00009: val_loss improved from 0.02005 to 0.01956, saving model to model-nvidia-v2-006.h5
Epoch 10/100
58/57 [==============================] - 53s 910ms/step - loss: 0.0212 - val_loss: 0.0202

Epoch 00010: val_loss did not improve from 0.01956
Epoch 11/100
58/57 [==============================] - 54s 925ms/step - loss: 0.0204 - val_loss: 0.0189

Epoch 00011: val_loss improved from 0.01956 to 0.01893, saving model to model-nvidia-v2-006.h5
Epoch 12/100
58/57 [==============================] - 53s 914ms/step - loss: 0.0199 - val_loss: 0.0188

Epoch 00012: val_loss improved from 0.01893 to 0.01875, saving model to model-nvidia-v2-006.h5
Epoch 13/100
58/57 [==============================] - 53s 915ms/step - loss: 0.0194 - val_loss: 0.0186

Epoch 00013: val_loss improved from 0.01875 to 0.01856, saving model to model-nvidia-v2-006.h5
Epoch 14/100
58/57 [==============================] - 52s 896ms/step - loss: 0.0189 - val_loss: 0.0188

Epoch 00014: val_loss did not improve from 0.01856
Epoch 15/100
58/57 [==============================] - 53s 922ms/step - loss: 0.0186 - val_loss: 0.0182

Epoch 00015: val_loss improved from 0.01856 to 0.01822, saving model to model-nvidia-v2-006.h5
Epoch 16/100
58/57 [==============================] - 53s 914ms/step - loss: 0.0181 - val_loss: 0.0183

Epoch 00016: val_loss did not improve from 0.01822
Epoch 17/100
58/57 [==============================] - 53s 910ms/step - loss: 0.0179 - val_loss: 0.0180

Epoch 00017: val_loss improved from 0.01822 to 0.01801, saving model to model-nvidia-v2-006.h5
Epoch 18/100
58/57 [==============================] - 53s 908ms/step - loss: 0.0177 - val_loss: 0.0181

Epoch 00018: val_loss did not improve from 0.01801
Epoch 19/100
58/57 [==============================] - 53s 917ms/step - loss: 0.0174 - val_loss: 0.0178

Epoch 00019: val_loss improved from 0.01801 to 0.01776, saving model to model-nvidia-v2-006.h5
Epoch 20/100
58/57 [==============================] - 53s 908ms/step - loss: 0.0170 - val_loss: 0.0178

Epoch 00020: val_loss did not improve from 0.01776
Epoch 21/100
58/57 [==============================] - 53s 919ms/step - loss: 0.0166 - val_loss: 0.0175

Epoch 00021: val_loss improved from 0.01776 to 0.01751, saving model to model-nvidia-v2-006.h5
Epoch 22/100
58/57 [==============================] - 53s 920ms/step - loss: 0.0165 - val_loss: 0.0174

Epoch 00022: val_loss improved from 0.01751 to 0.01740, saving model to model-nvidia-v2-006.h5
Epoch 23/100
58/57 [==============================] - 53s 922ms/step - loss: 0.0161 - val_loss: 0.0172

Epoch 00023: val_loss improved from 0.01740 to 0.01718, saving model to model-nvidia-v2-006.h5
Epoch 24/100
58/57 [==============================] - 53s 911ms/step - loss: 0.0157 - val_loss: 0.0168

Epoch 00024: val_loss improved from 0.01718 to 0.01681, saving model to model-nvidia-v2-006.h5
Epoch 25/100
58/57 [==============================] - 53s 914ms/step - loss: 0.0152 - val_loss: 0.0161

Epoch 00025: val_loss improved from 0.01681 to 0.01614, saving model to model-nvidia-v2-006.h5
Epoch 26/100
58/57 [==============================] - 52s 899ms/step - loss: 0.0149 - val_loss: 0.0149

Epoch 00026: val_loss improved from 0.01614 to 0.01495, saving model to model-nvidia-v2-006.h5
Epoch 27/100
58/57 [==============================] - 54s 927ms/step - loss: 0.0143 - val_loss: 0.0145

Epoch 00027: val_loss improved from 0.01495 to 0.01449, saving model to model-nvidia-v2-006.h5
Epoch 28/100
58/57 [==============================] - 53s 917ms/step - loss: 0.0135 - val_loss: 0.0138

Epoch 00028: val_loss improved from 0.01449 to 0.01382, saving model to model-nvidia-v2-006.h5
Epoch 29/100
58/57 [==============================] - 52s 905ms/step - loss: 0.0131 - val_loss: 0.0131

Epoch 00029: val_loss improved from 0.01382 to 0.01308, saving model to model-nvidia-v2-006.h5
Epoch 30/100
58/57 [==============================] - 53s 913ms/step - loss: 0.0128 - val_loss: 0.0132

Epoch 00030: val_loss did not improve from 0.01308
Epoch 31/100
58/57 [==============================] - 53s 915ms/step - loss: 0.0123 - val_loss: 0.0125

Epoch 00031: val_loss improved from 0.01308 to 0.01250, saving model to model-nvidia-v2-006.h5
Epoch 32/100
58/57 [==============================] - 53s 916ms/step - loss: 0.0120 - val_loss: 0.0127

Epoch 00032: val_loss did not improve from 0.01250
Epoch 33/100
58/57 [==============================] - 53s 913ms/step - loss: 0.0121 - val_loss: 0.0121

Epoch 00033: val_loss improved from 0.01250 to 0.01210, saving model to model-nvidia-v2-006.h5
Epoch 34/100
58/57 [==============================] - 53s 913ms/step - loss: 0.0117 - val_loss: 0.0121

Epoch 00034: val_loss did not improve from 0.01210
Epoch 35/100
58/57 [==============================] - 52s 897ms/step - loss: 0.0115 - val_loss: 0.0115

Epoch 00035: val_loss improved from 0.01210 to 0.01154, saving model to model-nvidia-v2-006.h5
Epoch 36/100
58/57 [==============================] - 53s 908ms/step - loss: 0.0114 - val_loss: 0.0116

Epoch 00036: val_loss did not improve from 0.01154
Epoch 37/100
58/57 [==============================] - 53s 915ms/step - loss: 0.0111 - val_loss: 0.0131

Epoch 00037: val_loss did not improve from 0.01154
Epoch 38/100
58/57 [==============================] - 54s 932ms/step - loss: 0.0112 - val_loss: 0.0119

Epoch 00038: val_loss did not improve from 0.01154
Epoch 39/100
58/57 [==============================] - 53s 912ms/step - loss: 0.0109 - val_loss: 0.0119

Epoch 00039: val_loss did not improve from 0.01154
Epoch 40/100
58/57 [==============================] - 53s 909ms/step - loss: 0.0109 - val_loss: 0.0113

Epoch 00040: val_loss improved from 0.01154 to 0.01131, saving model to model-nvidia-v2-006.h5
Epoch 41/100
58/57 [==============================] - 53s 909ms/step - loss: 0.0107 - val_loss: 0.0117

Epoch 00041: val_loss did not improve from 0.01131
Epoch 42/100
58/57 [==============================] - 53s 905ms/step - loss: 0.0106 - val_loss: 0.0127

Epoch 00042: val_loss did not improve from 0.01131
Epoch 43/100
58/57 [==============================] - 54s 928ms/step - loss: 0.0105 - val_loss: 0.0120

Epoch 00043: val_loss did not improve from 0.01131
Epoch 44/100
58/57 [==============================] - 54s 924ms/step - loss: 0.0104 - val_loss: 0.0122

Epoch 00044: val_loss did not improve from 0.01131
Epoch 45/100
58/57 [==============================] - 53s 918ms/step - loss: 0.0104 - val_loss: 0.0123

Epoch 00045: val_loss did not improve from 0.01131
Epoch 00045: early stopping
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive away from the edges of the road.

I then recorded the vehicle driving in the center lane in the opposite directio of the track.

Finally, I spent extra time recording the vehicle recovering from hitting the sides of bridges, and from turning corners where no barriers exist (the dirt roads).


To process the images I first imported the recorded driving log, and split these 80/20 between training and validation datasets.  I then created a generator to process each of the datasets as follows:

* shuffles
* loads the image, then converts to YUV color space
* adds a 0.2 angle offset for left camera images
* adds a -0.2 angle offset for right camera images
* flips the images (and angles)
* returns both the original and flipped images
  
The following shows 10 random images (the left, center and right camera angels), along with how they looked once they had been converted to YUV, normalized, cropped, and then resized.

![training-data-1][training-data-1]
![training-data-2][training-data-2]
![training-data-3][training-data-3]
![training-data-4][training-data-4]
![training-data-5][training-data-5]
![training-data-6][training-data-6]
![training-data-7][training-data-7]
![training-data-8][training-data-8]
![training-data-9][training-data-9]
![training-data-10][training-data-10]


### Simulation

#### 1. Navigating the car autonmously

Refer to the [video recording](video.mp4).