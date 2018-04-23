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
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
* training images were first read and loaded 
* HOG features were [extracted](https://github.com/verena-yacoub/SDCND-P5/blob/master/classifier_making.py#L39-L90) using `skimage.feature.hog()` function and different parameters were explored 
Below, is an example image of HOG feature extraction from a car and a noncar image, where the parameters are:  
orientation=11  
Pixels per cell=8  
Cells per bloack =2   
Colorspace= YUV  
Channels= ALL  
![alt text][image1]



#### 2. Explain how you settled on your final choice of HOG parameters.

In the following table all classifiers tried are grouped with their prediction accuracy

|Color space| 	Channels | HOG orientations|	Pixels per cell|	Cells per block|	Accuracy|
|:----------:|:-----------:|:--------------:|:--------------:|:---------|:---------:|
|HLS|	All|	11	|8	|2|	97.7%|
|HSV|	All|	11|	8	|2	|98.09%|
|YUV|	All|	11|	8|	2|	98.06%|
|YUV|	All|	11|	16|	2|	98%|
|YUV|	ALL|	9	|16|	2|	97.61%|
|YUV|	All|	9	|8	|2|	98.09%|

From experiments, all classifiers with ~ 97% accuracy are detecting more false positives and missing right spots. (YUV, ALL, 11, 8, 2) was chosen among other classifiers as it showed the most reasonable detection.  

*[here](https://github.com/verena-yacoub/SDCND-P5/tree/master/classifiers-ToMb) is a folder containing all built classifier models mentioned above*

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

A separate code was made [here](https://github.com/verena-yacoub/SDCND-P5/blob/master/classifier_making.py) to train and build SVM classifier model the this model is used in the main code. 
* In the code, the steps of classifier training and parameter tuning can be found [here](https://github.com/verena-yacoub/SDCND-P5/blob/master/classifier_making.py#L93-L142) 
* Note for normalization: in [this function](https://github.com/verena-yacoub/SDCND-P5/blob/master/classifier_making.py#L125) the default normalizer used by `skimage` library is the [L1-norm](http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog)  


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented over [here](https://github.com/verena-yacoub/SDCND-P5/blob/master/vehicle_detection.py#L94-L120) and combining multiple sliding window search can be found [here](https://github.com/verena-yacoub/SDCND-P5/blob/master/vehicle_detection.py#L173-L226)   
The main concept behind the scales chosen was that small cars are more likely in the top of the searching area while big sized cars can be captured in the bottom part of the image, so the start and end points are chosen accordingly 
  * another factor for choosing the end point in the Y axis is the scaling, as our original training image were 64x64 and that is the default sampling rate, it is important to have a suitable endpoint matching the searching scale, i.e. if the sclae is 3, the difference between the start and end points must be at least 3x64= 192.  
  
To visualize the sliding window search more: this image below shows the strips taken from the each image with the different scales, the figure is also shoeing the Hog feature of each  Y, U, V channel respectively beside. 

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

* Ultimately I searched on ten scales using YUV with orientations 11 and pixels per cells of 8. 
 * Noting that while training the classifier, the [default](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) penalty and loss parameters are kept, so, there was no special optimization for classifier training 

## Heat-map and filtering  
#### 1. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

* Positions of positive prediction are extracted by `nonzero` function  
* Then heatmap was [constructed](https://github.com/verena-yacoub/SDCND-P5/blob/master/vehicle_detection.py#L126-L140) 
* Threshold was then set to 3 (as I used many scales Threshold must be high, as the heatmap is representing roughly the number of positive prediction in this area, so when the value of heatmap is 4 for example, this means that 4 boxes gave positive signal and so on)
* Labels and boxes were [drawn](https://github.com/verena-yacoub/SDCND-P5/blob/master/vehicle_detection.py#L143-L158) 
* Another step of filtering is ignoring the thin bounding boxes which was implemented [here](https://github.com/verena-yacoub/SDCND-P5/blob/master/vehicle_detection.py#L154)


## Working pipeline  
the figure below is the complete working pipeline 

![alt text][image2]

---
### Video Implementation  
[Here](https://github.com/verena-yacoub/SDCND-P5/blob/master/project_output_video.mp4) is the link for the final video output!

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* In general, doing the balance between the scales and heatmap threshold needed many experiments, and though not avoiding all false positive prediction, and in order to avoid these in the submission heatmap threshold was set high to an extent of missing car when getting out of boxes' interference zone!  
* This pipeline might fail when dramatically varying brightness level
* As a way of improvment of quality, I guess changing the type of classifier, experimenting more with parameters, increasing the dataset for the classifier training (by augmentation or whole new images), or using history will do good, or else adding the color bins to the feature vector. 
* Knowing that quality improvement might be computationally intensive, I tried an [experiment](https://github.com/verena-yacoub/SDCND-P5/blob/master/experiment/experiment.py#L186-L214) where I reduced the scales of searching while keeping the same classifier and parameters, Resulting video can be found [here](https://github.com/verena-yacoub/SDCND-P5/blob/master/experiment/experiment_video_output.mp4) which seems promising and might be improved by enhancing the classifier!   
---
### References  
* Udacity classroom 
* https://github.com/jeremy-shannon/CarND-Vehicle-Detection/blob/master/vehicle_detection_project.ipynb
* https://github.com/abdelrahman-osama/CarND-Vehicle-Detection-P5/blob/master/Vehicle-Detection-P5.ipynb
