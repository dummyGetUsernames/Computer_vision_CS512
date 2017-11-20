#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:34:22 2017

@author: charusaxena
"""

import cv2

import numpy as np
import math
import sys 
 

#get image and thn gray it and float32 convertion
infile = sys.argv[1]
img = cv2.imread(infile)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

#detection of corners
distance = cv2.cornerHarris(gray,2,3,0.04)
distance = cv2.dilate(distance,None)
ret, distance = cv2.threshold(distance,0.01*distance.max(),255,0)
dst = np.uint8(distance)

ret, l, s, c = cv2.connectedComponentsWithStats(dst)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
detected_corners = cv2.cornerSubPix(gray,np.float32(c),(5,5),(-1,-1),criteria)

rows ,cols = detected_corners.shape 
sqrt1 = math.ceil(math.pow(rows,1/3))

obj = np.zeros((sqrt1 *sqrt1*sqrt1,3), np.float32)
obj[:,:3] = np.mgrid[0:sqrt1,0:sqrt1,0:sqrt1].T.reshape(-1,3)

infile = open('points_generated.txt','w')

for i,j in zip(obj,detected_corners): 
        X,Y,Z = i
        xi,yi = j.ravel()
        t = str(X) +" " + str(Y) + " "+str(Z) +" " + str(xi) + " " +str(yi)       
        infile.write(t+'\n')
        
print("a file is generated!")
infile.close()
cv2.destroyAllWindows()