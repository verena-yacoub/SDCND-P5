# SDCND-P5
## **Vehicle Detection Project**
---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./figures_for_MD/HOG_visualize.jpg
[image2]: ./figures_for_MD/processing_pipeline.jpg
[image3]: ./figures_for_MD/Channels_patches_HOG.jpg


## Rubric Points Discussion  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
* training images were first read and loaded 
* HOG features were extracted using `skimage.hog()` function and different parameters were explored 
Below, is an example image of HOG feature extraction from a car and a noncar image, where the parameters are:  
orientation=11  
Pixels per cell=8  
Cells per bloack =2   
Colorspace= YUV  
Channels= ALL
![alt text][image1]



#### 2. Explain how you settled on your final choice of HOG parameters.

In the following table all classifiers tried are grouped with their validation accuracy and comments 

|Color space| 	Channels | HOG orientations|	Pixels per cell|	Cells per block|	Accuracy|
|:----------:|:-----------:|:--------------:|:--------------:|:---------|:---------:|
|HLS|	All|	11	|8	|2|	97.7%|
|HSV|	All|	11|	8	|2	|98.09%|
|YUV|	All|	11|	8|	2|	98.06%|
|YUV|	All|	11|	16|	2|	98%|
|YUV|	ALL|	9	|16|	2|	97.61%|
|YUV|	All|	9	|8	|2|	98.09%|


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

A separate code was made [here]() to train and build SVM classifier model the this model is used in the main code. 
* In the code, the steps of classifier training and parameter tuning can be found [here]() 
* Note for normalization: in [this function]() the default normalizer used by `skimage` library is the [L1-norm](http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog)  


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented over [here] and combining multiple sliding window search can be found [here]()   
The main concept behind the scales chosen was that small cars are more likely in the top of the searching area while big sized cars can be captured in the bottom part of the image, so the start and end points are chosen accordingly 
  * another factor for choosing the end point in the Y axis is the scaling, as our original training image were 64x64 we are extracting...  
To visualize the sliding window search more: this image below shows the strips taken from the each image with the different scales, the image is also shoeing the Hog feature of each corresponding Y, U, V channels. 

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on seven scales using YUV with orientations 11 and pixels per cells of 8. 
 * Noting that while training the classifier, the [default](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) penalty and loss parameters are kept, so, there was no special optimization for classifier training 

## Heat-map and filtering  
#### 1. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 




---

### Video Implementation
[Here]() you can find the final video output!

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
