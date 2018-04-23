# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 12:55:41 2018

@author: Verena Maged
"""
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
import glob
import random
import numpy as np
from scipy.ndimage.measurements import label
from sklearn.externals import joblib

#---------------------------------------------------Import images and classifier --------------------------------------------------#
svc= joblib.load('classifier_YUV118_FVnon.pkl')

img = cv2.imread('test4.jpg')
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

car_images = glob.glob('trainingdata/vehicle/*.png')
noncar_images = glob.glob('trainingdata/non-vehicle/*.png')

car= cv2.imread(car_images[random.randint(0,len(car_images))])
car= cv2.cvtColor(car, cv2.COLOR_BGR2RGB)
carg= cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)

noncar= cv2.imread(noncar_images[random.randint(0,len(car_images))])
noncar= cv2.cvtColor(noncar, cv2.COLOR_BGR2RGB)
noncarg= cv2.cvtColor(noncar, cv2.COLOR_BGR2GRAY)



# --------------------------------- getting HOG features usinf skimage ---------------------------------------------------------------------------#

def get_hog_features(img, orient, pix_per_cell, cell_per_block,vis=True, feature_vec=True):
    
    if vis == True: # Call with two outputs if vis==True
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    
    else: # Otherwise call with one output     
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cell_per_block, cell_per_block), visualise=vis, feature_vector=feature_vec)
        return features

# --------------------------------- single function that can extract features using hog sub-sampling and make predictions------------------------#

def find_cars(img, ystart, ystop, scale, cspace, hog_channel, svc, orient, 
              pix_per_cell, cell_per_block, show_all_rectangles=True):
    
    allrectangles=[]
    rectangles = [] # array of rectangles where cars wil be detected
    
    img = img.astype(np.float32)/255 #normalizing image
    
    img_tosearch = img[ystart:ystop,:,:] #strip of image to search

    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(img)   
    
    # rescale image if other than 1.0 scale
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    # select colorspace channel for HOG 
    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    else: 
        ch1 = ctrans_tosearch[:,:,hog_channel]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)+1  #-1
    nyblocks = (ch1.shape[0] // pix_per_cell)+1  #-1 

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1_ = get_hog_features(ch1, orient, pix_per_cell, cell_per_block,feature_vec=False)
    hog1= hog1_ [0]
    if hog_channel == 'ALL':
        hog2_ = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3_ = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2= hog2_[0]
        hog3= hog3_[0]
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            
            if hog_channel == 'ALL':
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            
            hog_features=hog_features.reshape(1,-1)
            
            test_prediction = svc.predict(hog_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                rectangles.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
            
            if show_all_rectangles:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                allrectangles.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return rectangles,allrectangles, hog1_, hog2_, hog3_, ch1, ch2, ch3 

# --------------------------------- Draw boxes around labels ---------------------------------------------------------------------------#

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

# --------------------------------- Draw boxes around labels ---------------------------------------------------------------------------#
def draw_labeled_bboxes(img, labels):
    
    for car_number in range(1, labels[1]+1):# Iterate through all detected cars
        
        nonzero = (labels[0] == car_number).nonzero() # Find pixels' indices with each car_number label value
        
        nonzeroy = np.array(nonzero[0]) # Identify y values of those pixels
        nonzerox = np.array(nonzero[1]) # Identify x values of those pixels
        
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))) # Define a bounding box based on min/max x and y
        
        if (np.max(nonzerox)-np.min(nonzerox)<=55 and np.max(nonzeroy)-np.min(nonzeroy)>55) or (np.max(nonzeroy)-np.min(nonzeroy)<=55 and np.max(nonzerox)-np.min(nonzerox)>55) :
            continue
        else:
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)# Draw the box on the image
    return img

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    
    imcopy = np.copy(img)# Make a copy of the image
    random_color = False
   
    for bbox in bboxes: # Iterate through the bounding boxes
        if color == 'random' or random_color:
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            random_color = True
        
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)# Draw a rectangle given bbox coordinates
    return imcopy

#--------------------------------------------------------------------------------------------------------------------------#

rectangles = []
colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"


ystart = 400
ystop = 464
scale = 1.0
fun1= find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, orient, pix_per_cell, cell_per_block,show_all_rectangles=True)
rectangles.append(fun1[0])
test_all_rects= draw_boxes(img, fun1[1], color=(0, 0, 255))

ystart = 416
ystop = 480
scale = 1.0
fun2= find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, orient, pix_per_cell, cell_per_block,show_all_rectangles=True)
rectangles.append(fun2[0])
test_all_rects= draw_boxes(test_all_rects, fun2[1], color=(255, 0, 0))

ystart = 400
ystop = 496
scale = 1.5
fun3= find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, orient, pix_per_cell, cell_per_block,show_all_rectangles=True)
rectangles.append(fun3[0])
test_all_rects= draw_boxes(test_all_rects, fun3[1], color=(0, 255, 0))


ystart = 432
ystop = 528
scale = 1.5
fun4= find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, orient, pix_per_cell, cell_per_block,show_all_rectangles=True)
rectangles.append(fun4[0])
test_all_rects= draw_boxes(test_all_rects, fun4[1], color=(255, 255, 255))

ystart = 400
ystop = 528
scale = 2.0
fun5= find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, orient, pix_per_cell, cell_per_block,show_all_rectangles=True)
rectangles.append(fun5[0])
test_all_rects= draw_boxes(test_all_rects, fun5[1], color=(255, 0, 255))

ystart = 432
ystop = 560
scale = 2.0
fun6= find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, orient, pix_per_cell, cell_per_block,show_all_rectangles=True)
rectangles.append(fun6[0])
test_all_rects= draw_boxes(test_all_rects, fun6[1], color=(255, 0, 255))

ystart = 400
ystop = 592
scale = 3
fun7= find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, orient, pix_per_cell, cell_per_block,show_all_rectangles=True)
rectangles.append(fun7[0])
test_all_rects= draw_boxes(test_all_rects, fun7[1], color=(0, 255, 255))

ystart = 450
ystop = 642
scale = 3
fun8= find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, orient, pix_per_cell, cell_per_block,show_all_rectangles=True)
rectangles.append(fun8[0])
test_all_rects= draw_boxes(test_all_rects, fun8[1], color=(0, 255, 255))

ystart = 400
ystop = 596
scale = 3.5
fun9= find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, orient, pix_per_cell, cell_per_block,show_all_rectangles=True)
rectangles.append(fun9[0])
test_all_rects= draw_boxes(test_all_rects, fun9[1], color=(0, 255, 255))


ystart = 456
ystop = 680
scale = 3.5
fun10= find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, orient, pix_per_cell, cell_per_block,show_all_rectangles=True)
rectangles.append(fun10[0])
test_all_rects= draw_boxes(test_all_rects, fun10[1], color=(0, 255, 255))

# apparently this is the best way to flatten a list of lists
rectangles = [item for sublist in rectangles for item in sublist] 

test_img_rects = draw_boxes(img, rectangles)

heatmap_img = np.zeros_like(img[:,:,0])
heatmap_img = add_heat(heatmap_img, rectangles)
#plt.figure(figsize=(10,10))
#plt.imshow(heatmap_img, cmap='hot')


heatmap_img = apply_threshold(heatmap_img, 3)
#plt.figure(figsize=(10,10))
#plt.imshow(heatmap_img, cmap='hot')

labels = label(heatmap_img)
#plt.figure(figsize=(10,10))
#plt.imshow(labels[0], cmap='gray')


# Draw bounding boxes on a copy of the image
draw_img = draw_labeled_bboxes(np.copy(img), labels)

    
features= get_hog_features(carg, 11,8, 2, vis=True, feature_vec=False)
features1= get_hog_features(noncarg, 11,8, 2, vis=True, feature_vec=False)

f, ([ax1,ax2,ax3,ax4]) = plt.subplots(1,4, figsize=(15,15))
ax1.imshow(car)
ax1.set_title('car image', fontsize=20) 
ax2.imshow(features[1], cmap='gray')
ax2.set_title('car HOG', fontsize=20)
ax3.imshow(noncar)
ax3.set_title('noncar image', fontsize=20) 
ax4.imshow(features1[1], cmap='gray')
ax4.set_title('noncar HOG', fontsize=20) 
plt.savefig('HOG_visualize.jpg')

f2, ([ax5,ax6,ax7,ax8]) = plt.subplots(1,4, figsize=(20,20))
ax5.imshow(test_all_rects)
ax5.set_title('all rectangles', fontsize=20) 
ax6.imshow(test_img_rects)
ax6.set_title('rectangles with positive prediction', fontsize=15) 
ax8.imshow (draw_img)
ax8.set_title('final view', fontsize=20) 
ax7.imshow (heatmap_img, cmap='hot')
ax7.set_title('Applying heatmap', fontsize=20) 
plt.savefig('processing_pipeline.jpg')


f3, ([ax9,ax10,ax11,ax12,ax13,ax14],[ax15,ax16,ax17,ax18,ax19,ax20],[ax21,ax22,ax23,ax24,ax25,ax26],[ax27,ax28,ax29,ax30,ax31,ax32],[ax33,ax34,ax35,ax36,ax37,ax38],[ax39,ax40,ax41,ax42,ax43,ax44],[ax45,ax46,ax47,ax48,ax49,ax50],[ax51,ax52,ax53,ax54,ax55,ax56],[ax57,ax58,ax59,ax60,ax61,ax62],[ax63,ax64,ax65,ax66,ax67,ax68]) = plt.subplots(10,6, figsize=(15,5))



ax9.imshow (fun1[5])
ax9.axis('off')
ax10.imshow (fun1[2][1], cmap='gray')
ax10.axis('off')
ax11.imshow (fun1[6])
ax11.axis('off')
ax12.imshow (fun1[3][1], cmap='gray')
ax12.axis('off')
ax13.imshow (fun1[7])
ax13.axis('off')
ax14.imshow (fun1[4][1], cmap='gray')
ax14.axis('off')

ax15.imshow (fun2[5])
ax15.axis('off')
ax16.imshow (fun2[2][1], cmap='gray')
ax16.axis('off')
ax17.imshow (fun2[6])
ax17.axis('off')
ax18.imshow (fun2[3][1], cmap='gray')
ax18.axis('off')
ax19.imshow (fun2[7])
ax19.axis('off')
ax20.imshow (fun2[4][1], cmap='gray')
ax20.axis('off')

ax21.imshow (fun3[5])
ax21.axis('off')
ax22.imshow (fun3[2][1], cmap='gray')
ax22.axis('off')
ax23.imshow (fun3[6])
ax23.axis('off')
ax24.imshow (fun3[3][1], cmap='gray')
ax24.axis('off')
ax25.imshow (fun3[7])
ax25.axis('off')
ax26.imshow (fun3[4][1], cmap='gray')
ax26.axis('off')


ax27.imshow (fun4[5])
ax27.axis('off')
ax28.imshow (fun4[2][1], cmap='gray')
ax28.axis('off')
ax29.imshow (fun4[6])
ax29.axis('off')
ax30.imshow (fun4[3][1], cmap='gray')
ax30.axis('off')
ax31.imshow (fun4[7])
ax31.axis('off')
ax32.imshow (fun4[4][1], cmap='gray')
ax32.axis('off')

ax33.imshow (fun5[5])
ax33.axis('off')
ax34.imshow (fun5[2][1], cmap='gray')
ax34.axis('off')
ax35.imshow (fun5[6])
ax35.axis('off')
ax36.imshow (fun5[3][1], cmap='gray')
ax36.axis('off')
ax37.imshow (fun5[7])
ax37.axis('off')
ax38.imshow (fun5[4][1], cmap='gray')
ax38.axis('off')


ax39.imshow (fun6[5])
ax39.axis('off')
ax40.imshow (fun6[2][1], cmap='gray')
ax40.axis('off')
ax41.imshow (fun6[6])
ax41.axis('off')
ax42.imshow (fun6[3][1], cmap='gray')
ax42.axis('off')
ax43.imshow (fun6[7])
ax43.axis('off')
ax44.imshow (fun6[4][1], cmap='gray')
ax44.axis('off')

ax45.imshow (fun3[5])
ax45.axis('off')
ax46.imshow (fun3[2][1], cmap='gray')
ax46.axis('off')
ax47.imshow (fun3[6])
ax47.axis('off')
ax48.imshow (fun3[3][1], cmap='gray')
ax48.axis('off')
ax49.imshow (fun3[7])
ax49.axis('off')
ax50.imshow (fun3[4][1], cmap='gray')
ax50.axis('off')


ax51.imshow (fun4[5])
ax51.axis('off')
ax52.imshow (fun4[2][1], cmap='gray')
ax52.axis('off')
ax53.imshow (fun4[6])
ax53.axis('off')
ax54.imshow (fun4[3][1], cmap='gray')
ax54.axis('off')
ax55.imshow (fun4[7])
ax55.axis('off')
ax56.imshow (fun4[4][1], cmap='gray')
ax56.axis('off')

ax57.imshow (fun5[5])
ax57.axis('off')
ax58.imshow (fun5[2][1], cmap='gray')
ax58.axis('off')
ax59.imshow (fun5[6])
ax59.axis('off')
ax60.imshow (fun5[3][1], cmap='gray')
ax60.axis('off')
ax61.imshow (fun5[7])
ax61.axis('off')
ax62.imshow (fun5[4][1], cmap='gray')
ax62.axis('off')


ax63.imshow (fun6[5])
ax63.axis('off')
ax64.imshow (fun6[2][1], cmap='gray')
ax64.axis('off')
ax65.imshow (fun6[6])
ax65.axis('off')
ax66.imshow (fun6[3][1], cmap='gray')
ax66.axis('off')
ax67.imshow (fun6[7])
ax67.axis('off')
ax68.imshow (fun6[4][1], cmap='gray')
ax68.axis('off')



plt.savefig('Channels_patches_HOG.jpg')
