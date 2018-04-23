# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 21:59:36 2018

@author: Verena Maged
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 07:46:38 2018

@author: Verena Maged
"""

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
from sklearn.externals import joblib
import cv2
import glob
import time

#-------------------------------- Load the training or test data -------------------------------------------------------------------#
car_images = glob.glob('trainingdata/vehicle/*.png')
noncar_images = glob.glob('trainingdata/non-vehicle/*.png')
print(len(car_images), len(noncar_images))

#-------------------------------- Get HOG features -------------------------------------------------------------------#

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=True, 
                     feature_vec=True):
    return_list = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cell_per_block, cell_per_block), visualise= vis, feature_vector= feature_vec)
    
    hog_features = return_list[0]
    if vis:
        hog_image = return_list[1]
        return hog_features, hog_image
    else:
        return hog_features
    
#-------------------------------- extract features from dataset -------------------------------------------------------------------#
    
def extract_features1(imgs, cspace='RGB', orient=9,pix_per_cell=8, cell_per_block=2, hog_channel=0):
    
    features = []# Create a list to append feature vectors to
    
    for file in imgs:# Iterate through the list of images
        
        image = mpimg.imread(file) # Read in each one by one noting that this function is reading in RGB order
        
        
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                feat=get_hog_features(feature_image[:,:,channel], orient,pix_per_cell, cell_per_block, vis=True, feature_vec=False)
               
                hog_features.append(feat[0])
            
            hog_features = np.ravel(hog_features) 
            
        else:
            hog_features, visualize = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
        
        features.append(hog_features) # Append the new feature vector to the features list
    return features

#-------------------------------- Choosing parameters -------------------------------------------------------------------#

colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

t1=time.time()
car_features1 = extract_features1(car_images, cspace=colorspace, orient=orient,pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
notcar_features1 = extract_features1(noncar_images, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
t21 = time.time()
print(round(t21-t1, 2), 'Seconds to extract HOG features...')


X1 = np.vstack((car_features1, notcar_features1)).astype(np.float64) # Create an array stack of feature vectors


y1 = np.hstack((np.ones(len(car_features1)), np.zeros(len(notcar_features1)))) # Define the labels vector


rand_state1 = np.random.randint(0, 100) 
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=rand_state1) # Split up data into randomized training and test sets
    

X_scaler1 = StandardScaler().fit(X_train1) # Fit a per-column scaler

X_train1 = X_scaler1.transform(X_train1) # Apply the scaler to X
X_test1 = X_scaler1.transform(X_test1)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train1[0]))

svc1 = LinearSVC()# Use a linear SVC 

#----------------------------------------- Check the training time for the SVC-----------------------------------------------#
t1=time.time()
svc1.fit(X_train1, y_train1)
t21 = time.time()
print(round(t21-t1, 2), 'Seconds to train SVC...')

print('Test Accuracy of SVC = ', round(svc1.score(X_test1, y_test1), 4))# Check the score of the SVC


#----------------------------------------- Check the prediction time for the SVC-----------------------------------------------#
t1=time.time()
n_predict = 10
print('My SVC predicts: ', svc1.predict(X_test1[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test1[0:n_predict])
t21 = time.time()
print(round(t21-t1, 5), 'Seconds to predict', n_predict,'labels with SVC')

joblib.dump(svc1, 'classifier-YUV11-8.pkl') # save the model 