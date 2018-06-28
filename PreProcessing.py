# -*- coding: utf-8 -*-
# @Author: Yan Tang
# @Date:   2018-06-27 

'''
----------------------------------------------------------------
Load data for '.pkl' file. 
Generate differential geometric data.
Convert label to one hot label. 
For example: 
3->[0,0,1,0,0,0] and 5->[0,0,0,0,1,0].
Load data for model training for cnn_for_fera_ten_fold_ten.py
----------------------------------------------------------------
'''

import os
import pickle 
import cv2

def pickle_2_img(data_file):
     if not os.path.exists(data_file):
        print('file {0} not exists'.format(data_file))
        exit()
     with open(data_file, 'rb') as f:
        data = pickle.load(f)
     total_x1, total_x2, total_x3, total_gx, total_y = [], [], [], [], []
     #ten groups data for ten-fold cross validation
     for i in range(len(data)):         
         x1 = []
         x2 = []
         x3 = []
         x4 = []
         yl = []
         print(len(data[i]['img']))
         for j in range(len(data[i]['labels'])):
             geo_array = data[i]['geometry'][j]
             v0 = geo_array[0]
             v1 = geo_array[1]
             v2 = geo_array[2]
             img_array = data[i]['img'][j]

             #the first image
             img1 = img_array[0]
             img1 = img1.flatten()          
             #the middle image
             img2 = img_array[1]
             img2 = img2.flatten()
             #the last image
             img3 = img_array[2]
             img3 = img3.flatten()
             
             #final difference
             v = list(map(lambda x: x[0]-x[1], zip(v2, v0))) 
             #dynamicn geometric feature
             gx = v2+v

             label = int(data[i]['labels'][j][2])
             
             #label mapping
             if label==7:
                 label = 2
                              
             label = dense_to_one_hot(label,6)
                          
             x1.append(img1)
             x2.append(img2)
             x3.append(img3)
             x4.append(gx)
             yl.append(label)

         total_x1.append(x1)
         total_x2.append(x2)
         total_x3.append(x3)
         total_gx.append(x4)
         total_y.append(yl)    
           
     return total_x1, total_x2, total_x3, total_gx, total_y

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = []
    for i in range(num_classes):
        if i==labels_dense-1:
            labels_one_hot.append(1)
        else:
            labels_one_hot.append(0)
    return labels_one_hot
