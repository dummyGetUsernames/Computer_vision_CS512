#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 19:23:50 2017

@author: charusaxena231
"""

import cv2

import numpy as np
#from matplotlib import pyplot as plt
import math 
import time
while(1):
    cap = cv2.VideoCapture(0)
    ret,img = cap.read()
    cap.release()
   
    time.sleep(3)     
   
       
   
    cap = cv2.VideoCapture(0)
    ret,second = cap.read()
    cap.release() 

    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    
    def HarrisHandler(i):
        n=cv2.getTrackbarPos('threshold','frame')
        k = cv2.getTrackbarPos('kvalue','frame') 
        w = cv2.getTrackbarPos('windowsize','frame') 
        if n==0:
             n= 0.5
        else:
             n = (n/10)*1
        
        if k==0:
            k==0.01
        else:
            k = (k/100)*2
        print("k value",k)  
        
        if w==0:
            w=3
        
#==============================================================================
#         
#==============================================================================
         
        eigen = cv2.cornerEigenValsAndVecs(gray, w, 3)
        mc= np.zeros(gray.shape)
    
        rows,cols = gray.shape
#==============================================================================
        for i in range(rows):
            for j in range(cols):
                lam1 = eigen[i,j][0]
                lam2 = eigen[i,j][1]
                mc[i,j]=(lam1*lam2)-(k*(math.pow((lam1+lam2),2)))
#==============================================================================
       

        
        corner = []
        minvalue,maxvalue,minloc,maxloc=cv2.minMaxLoc(mc)
        threshold = n * maxvalue
        
        for index , x in np.ndenumerate(mc):
                if x > threshold:
                    corner.append(index)
                        
    #normalise
        
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 0.1)
        total_corners = np.float32(corner)
        
        cv2.cornerSubPix( gray, total_corners, (5,5), (-1,-1), criteria )
        
#==============================================================================
        uniqueCorners = np.unique(np.int0(total_corners),axis=0)
        

        flag=1
        img1 = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
#==============================================================================
        #print(type(uniqueCorners))
        for i in uniqueCorners:
            x,y = i.ravel()
            cv2.rectangle(img1, (y-10,x-10), (y+10,x+10), (0,255,0)) 
            cv2.putText(img1,str(flag),(y,x),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
            flag+=1
        kp1 = [cv2.KeyPoint(x[1], x[0], 3) for x in uniqueCorners]
        
        second_image_processing(threshold,k,img1,kp1,w)
        
    #def orb(img1,second1): 
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def second_image_processing(thresh,k_value,img1,kp1,w):
        gray2= cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
        
        eigen = cv2.cornerEigenValsAndVecs(gray2, w, 3)
        print(type(eigen))
        mc1= np.zeros(gray2.shape)
    
        rows,cols = gray2.shape
        for i in range(rows):
            for j in range(cols):
                lam1 = eigen[i,j][0]
                lam2 = eigen[i,j][1]
                mc1[i,j] =(lam1*lam2)-(k_value*(math.pow((lam1+lam2),2)))
                    
    
        
        corner2 = []
        minvalue,maxvalue,minloc,maxloc=cv2.minMaxLoc(mc1)
             
        
#==============================================================================
#=============================================================================
        for index , x in np.ndenumerate(mc1):
                if x > thresh:
                    corner2.append(index)
    #normalise
        
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1)
        total_corners1 = np.float32(corner2)
        
        cv2.cornerSubPix( gray2, total_corners1, (5,5), (-1,-1), criteria )
        uniqueCorners1 = np.unique(np.int0(total_corners1),axis=0)
     
        
        print("unique",len(uniqueCorners1))
        flag=1
        second1 = cv2.cvtColor(gray2,cv2.COLOR_GRAY2BGR)
        print(flag)
        for i in uniqueCorners1:
            x,y = i.ravel()
            cv2.rectangle(second1, (y-10,x-10), (y+10,x+10), (0,255,0)) 
            cv2.putText(second1,str(flag),(y,x),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
            flag+=1
        kp2 = [cv2.KeyPoint(x[1], x[0], 3) for x in uniqueCorners1]
        
        feature_matching(img1,second1,kp1,kp2)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
       
    def feature_matching(img1,second1,kp1,kp2):
        
        orb = cv2.ORB_create()
        kp3,des1 = orb.compute(img1,kp1)
        kp4,des2 = orb.compute(second1,kp2)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
        matches = bf.match(des1,des2)
   
        matches = sorted(matches, key = lambda x:x.distance)
        print(kp3)
        img3 = cv2.drawMatches(img1,kp3,second1,kp4,matches,None,flags=2)
        cv2.destroyWindow("img")
        cv2.imshow("img",img3)
        
    
    #harrisHandler(img,img2,gray)
    cv2.imshow("frame",img)
    cv2.createTrackbar('threshold','frame',0,5,HarrisHandler)
    cv2.createTrackbar('kvalue','frame',0,8,HarrisHandler)
    cv2.createTrackbar('windowsize','frame',0,8,HarrisHandler)
    
    
    k = cv2.waitKey(0)
    if k ==27:
        cv2.destroyAllWindows()
        break
    elif k == ord('q'):
        cv2.destroyAllWindows()
       
