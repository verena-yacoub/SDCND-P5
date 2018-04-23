# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 11:13:32 2018

@author: Verena Maged
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 10:59:19 2018

@author: Verena Maged
"""


"""
Created on Sat Mar 24 13:30:54 2018

@author: Verena Yacoub
"""



from skimage.feature import hog
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import numpy as np
from sklearn.externals import joblib
import cv2
import matplotlib.image as mpimg

svc= joblib.load('classifier_YUV118_FVnon.pkl')


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
              pix_per_cell, cell_per_block, show_all_rectangles=False):
    
    
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
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block,feature_vec=False)
    hog1= hog1 [0]
    if hog_channel == 'ALL':
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2= hog2[0]
        hog3= hog3[0]
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
                
    return rectangles 

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
        
        if (np.max(nonzerox)-np.min(nonzerox)<=55 ) or (np.max(nonzeroy)-np.min(nonzeroy)<=55) :
            continue
        else:
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)# Draw the box on the image
    return img

    
# --------------------------------- Pipeline ---------------------------------------------------------------------------#
def process_frame_for_video(img): # pipline for each frame
        
        rectangles = []
        
        colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 11
        pix_per_cell = 8
        cell_per_block = 2
        hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
        
        ystart = 400
        ystop = 496
        scale = 1.5
        rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, orient, pix_per_cell, cell_per_block))

        ystart = 432
        ystop = 528
        scale = 1.5
        rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, orient, pix_per_cell, cell_per_block))

        ystart = 400
        ystop = 528
        scale = 2.0
        rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, orient, pix_per_cell, cell_per_block))

        ystart = 432
        ystop = 560
        scale = 2.0
        rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, orient, pix_per_cell, cell_per_block))



 
        rectangles = [item for sublist in rectangles for item in sublist] #flatten a list of lists
        
        heatmap_img = np.zeros_like(img[:,:,0]) # creating a blank image
        heatmap_img = add_heat(heatmap_img, rectangles) #adding heat each Bbox is counted 1
        
        heatmap_img = apply_threshold(heatmap_img, 1) # thresholding heatmap
        
        labels = label(heatmap_img) # extract lables from thresholded heatmap 
        
        draw_img = draw_labeled_bboxes(np.copy(img), labels)# Draw bounding boxes on a copy of the image
        return  draw_img
   


test_out_file2 = 'madness2output.mp4'
clip_test2 = VideoFileClip('project_video.mp4')
clip_test_out2 = clip_test2.fl_image(process_frame_for_video)#.subclip(41,43)
clip_test_out2.write_videofile(test_out_file2, audio=False)