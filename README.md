# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_image/dataset_example.png "Visualization"
[image2]: ./output_image/grayscale.png "Grayscaling"
[image3]: ./output_image/norm.png "Normalized"
[image4]: ./output_image/bar_chart.png "Dataset STATS"
[image5]: ./output_image/0.png "Traffic Sign 1"
[image6]: ./output_image/1.png "Traffic Sign 2"
[image7]: ./output_image/2.png "Traffic Sign 3"
[image8]: ./output_image/3.png "Traffic Sign 4"
[image9]: ./output_image/4.png "Traffic Sign 5"
[image10]: ./output_image/5.png "Traffic Sign 6"
[image11]: ./output_image/6.png "Traffic Sign 7"
[image12]: ./output_image/7.png "Traffic Sign 8"
[image13]: ./output_image/8.png "Traffic Sign 9"
[image14]: ./output_image/9.png "Traffic Sign 10"
---

### Data Set Summary & Exploration

#### 1. A basic summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. An exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The first image is some of the traffic sign classes. They are the Speed Limit (50km/h) Sign, Prority Road Sign, Speed Limit (70km/h) Sign, Bicycles Crossing. And the second image is a bar chart showing how the data is distributed.

![example][image1]
![bar chart][image4]

### Design and Test a Model Architecture

#### 1. Data Preprocessing 

As a first step, I decided to convert the images to grayscale because comparing to a RGB image, a grayscale image only have 1 channel. It contains less color related information and easier for the network to learn the image's feature. 

Here is an example of a traffic sign image before and after grayscaling.

![before][image1]
![after][image2]

As a last step, I normalized the image data because in this way the data has mean zero and equal variance. 

![normalized][image3]

#### 2. Final model architecture.

The model is adapted from the Lenet-5 architecture. I added more dropout layer and a fully-connected layer.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray-scale image  					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU 			    	|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16    				|
| Flatten   			| inputs 5x5x16 outputs 400  					|
| Dropout				|												|
| Fully connected		| inputs 400 outputs 200  						|
| RELU 			    	|												|
| Dropout				|												|
| Fully connected		| inputs 200 outputs 100  						|
| RELU 			    	|												|
| Dropout				|												|
| Fully connected		| inputs 100 outputs 86  						|
| Fully connected		| inputs 86 outputs 43  						|
|						|												|
|						|												|
 


#### 3. Model Training. 

To train the model, I used cross entropy and Adam optimizer. Because the Adam optimizer usually works well with little hyperparameter tuning. 

As for hyperparameter, I set epoch = 30, batch_size = 128, and learning rate = 0.0005 as the final solution. 

As first, I used epoch = 10, batch_size = 64, and learning rate = 0.001. However, the validation accuracy stopped at 78%-80%. I thought, maybe the learning rate was too high and the model missed some important features while finding the best solution. So, I tried decreasing the learning rate to 0.0007 and 0.0005. However, the validation accuracy didn't increased that much as I expected.

Then I tried to increased the batch_size to 128 and 256, which were 2 times and 4 times of 64. Unfortunately, the validation accuracy was still less than 93%. Sometimes, the validation accuracy became lower than before. I searched online and found: Larger batch_size results in faster training  progress, but may not converge as fast. Smaller batch_size trains slower, but can converge faster. Also, bigger number of epoches can lead to higher accuracy. So, I decided to decrased the batch_size to 128 and tried increasing # of epoch.

I tried with epoch = 12, epoch = 24, and epoch = 30. The best model (highest validation accuracy) was trained when the epoch = 30. That was how I get the hyperparameter combination.

#### 4. Model Evaluation

Along with hyperparameter tuning, I also tried to modified the layers in model architecture.

As fist, I used average pooling and max pooling. Both of these pooling methods reduced half of the output size. I compared this combination (1 average pooling layer and 1 max pooling layer) with 2 max pooling layers combination. The 2 max pooling layers combination received a higher validaiton accuracy. 

About the fully connected (FC) layer, the original Lenet-5 architecture has 3 FC layers. But the Lenet-5 model only used to classify number 0-9, 10 classes. In traffic sign classification problem, there were 43 classes. To I added another FC layer to adapted the 43 classes and 400 outputs (5x5x16 = 400) after flatten.

Also, at the very begining, I only used 1 dropout layer after flatten. Then I realized the dropout layer can help prevent overfitting and I added more FC layers, which may cause the model to overfit. So, I added more dropout layers after FC layers to further prevent overfitting.

My final model results were:
* training set accuracy of 98.1%
* validation set accuracy of 93.7% 
* test set accuracy of 91.6%

The validation set accuracy was greater than 93% but only 0.7% greater than 93%. And the test set accuracy was 91.6%, which was not very high. I believed both the validaiton set accuracy and test set accuracy could grow higher. Maybe because the added dropout layers cause negative impact to the model.

### Test a Model on New Images

#### 1. Test on German traffic signs dataset found on the web.

In total, I chose 10 images from the German traffic sign dataset. Here are the 10 German traffic signs images:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10]
![alt text][image11] ![alt text][image12] ![alt text][image13]
![alt text][image14]

The first image might be difficult to classify because the image itself is small.

The second image might be difficult to classify because the image is small and the shape of the sign becomes oval. 

The third image might be difficult to classify because the image is small and the boundry of the sign is not very clear. 

The fourth image might be difficult to classify because the image is small and the sign is pretty blurry.

The fifth image might be difficult to classify because the image itself is small.

The sixth image might be difficult to classify because the image is small and the boundry of the sign is not very clear.

The seventh image might be difficult to classify because the image itself is small.

The eighth image might be difficult to classify because the image is small and the sign is pretty blurry.

The nineth image might be difficult to classify because the image is small and the sign is pretty blurry.

The tenth image might be difficult to classify because the image is small and the shape of the sign becomes oval. Also, there is another sign in the back, overlapped with the target sign. This might create additional difficulties.

#### 2. Model Prediction.

In order for the model to make prediction, I resized the original images to 32x32 and converted to gray-scale and normalized the images.

Here are the results of the prediction:

| Image   			         |     Prediction	        					| 
|:--------------------------:|:--------------------------------------------:| 
| 3.5 tons vehicle prohibited| 3.5 tons vehicle prohibited					| 
| 30 km/h     				 | 30 km/h 										|
| Keep Right				 | Keep Right									|
| Turn Right Ahead			 | Turn Right Ahead								|
| Right at next interseation | Right at next interseation   				| 
| Keep Right     			 | Keep Right  									|
| General Caution			 | General Caution								|
| Priority Road 	 		 | Priority Road				 				|
| Road Work					 | Road Work      								|
| Ahead Only				 | Ahead Only					 				|

The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 91.6%

#### 3. Top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 22nd cell of the Ipython notebook.

For the 1st image, the model is sure that this is a prohibited sign (probability of 99.98%), and the image does contain a sprohibited sign. The top five soft max probabilities were

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| 99.98%      			| Prohibited sign   					| 
| 0.02%     			| No Passing 							|
| 0.00%					| Roundabout Mandatory					|
| 0.00%	      			| No Passing Sign						|
| 0.00%				    | Speed Limit 100km/h      				|


For the 2nd image, the model is sure that this is a speed limit sign (probability of 99.94%), and the image does contain a speed limit sign. The top five soft max probabilities were

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| 99.94%      			| Speed Limit 30km/h      				| 
| 0.05%     			| Speed Limit 20km/h      				| 
| 0.00%					| Speed Limit 50km/h      				| 
| 0.00%	      			| Speed Limit 70km/h      				| 
| 0.00%				    | Speed Limit 80km/h      				| 
 

For the 3rd image, the model is sure that this is a keep right sign (probability of 100%), and the image does contain a keep right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| 100%      			| Keep Right      						| 
| 0.00%     			| Turn Left Ahead 						| 
| 0.00%					| Yield      							| 
| 0.00%	      			| Road Work      						| 
| 0.00%				    | Priority Road      					|


For the 4th image, the model is sure that this is a turn right ahead sign (probability of 99.82%), and the image does contain a sturn right ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| 99.82%      			| Turn Right Ahead      				| 
| 0.16%     			| Stop Sign      						| 
| 0.01%					| Ahead Only      						| 
| 0.00%	      			| No Entry      						| 
| 0.00%				    | No Vehicles      						|


For the 5th image, the model is sure that this is a intersection sign (probability of 99.55%), and the image does contain a intersection sign. The top five soft max probabilities were

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| 99.55%      			| Intersection Sign      				| 
| 0.41%     			| Beware Sign      						| 
| 0.03%					| Double Curve      					| 
| 0.00%	      			| Pedestrians      						| 
| 0.00%				    | Road Narrows      					|


For the 6th image, the model is sure that this is a keep right sign (probability of 100%), and the image does contain a keep right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| 100%      			| Keep Right      						| 
| 0.00%     			| Turn Left Ahead 						| 
| 0.00%					| Yield      							| 
| 0.00%	      			| Road Work      						| 
| 0.00%				    | Priority Road      					|

Also, the top 5 probabilities for the 6th image are the same as the 3rd image. Here, we can conclude the model does have consistency.


For the 7th image, the model is sure that this is a general caution sign (probability of 99.90%), and the image does contain a general caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| 99.90%      			| General Caution      					| 
| 0.10%     			| Traffic Sign      					| 
| 0.00%					| Pedestrians      						| 
| 0.00%	      			| Road Narrows      					| 
| 0.00%				    | Intersection      					|


For the 8th image, the model is sure that this is a priority road sign (probability of 99.98%), and the image does contain a priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| 99.98%      			| Priority Road      					| 
| 0.01%     			| Roundabout Mandatory  				| 
| 0.00%					| No Entry      						| 
| 0.00%	      			| Stop Sign      						| 
| 0.00%				    | Yield      							|


For the 9th image, the model is sure that this is a road work sign (probability of 100%), and the image does contain a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| 100%      			| Road Work      						| 
| 0.00%     			| Bumpy Road      						| 
| 0.00%					| Traffic Signals      					| 
| 0.00%	      			| General Caution      					| 
| 0.00%				    | Road Narrows      					|


For the 10th image, the model is sure that this is a ahead only sign (probability of 100%), and the image does contain a ahead only sign. The top five soft max probabilities were

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| 100%      			| Ahead Only      						| 
| 0.00%     			| Turn Left Ahead      					| 
| 0.00%					| Speed Limit 60km/h      				| 
| 0.00%	      			| Yield      							| 
| 0.00%				    | Turn Right Ahead      				|
