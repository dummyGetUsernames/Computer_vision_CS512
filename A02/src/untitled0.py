#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 22:01:08 2017
@author: charusaxena
"""
import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys


if len(sys.argv) == 2:
     img0 = sys.argv[1]
elif len(sys.argv)<2 :
     cap=cv.VideoCapture(0)
     a,img0= cap.read()
img=img0
cap.release()


cv.imshow('image',img)
b,g,r = cv.split(img)
x=1
imge = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
r,c,a = img.shape

def gray():
    imge_copy= img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            average = int(sum(img[i,j])/3)
            imge_copy[i,j] = (average,average,average)
    cv.imshow('image',img)
def slideHandler(i):
    global img
    n=cv.getTrackbarPos('trackbar','image')
    
    img = cv.filter2D(imge,-1)
    cv.imshow('image',img)

def rotateHandler(i):
    global img
    n=cv.getRotationMatrix2D((r/2,c/2),i,1)
    img = cv.warpAffine(imge,n,(c,r))
    cv.imshow('image',img)


while(1):
    k = cv.waitKey(0)
    if k == 27:
        break
    
    elif k == ord('i'):
        img=img0
        cv.imshow('image',img)
        
    elif k == ord('w'):
        cv.imwrite('out.jpg',img)
        
    elif k == ord('g'):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
        cv.imshow('image',img)
    
    elif k == ord('G'):
        gray
        
    elif k == ord('c'):
        if x==1:
            img=b
            cv.imshow('image',img)
            x=2
        elif x==2:
            img=g
            cv.imshow('image',img)
            x=3
        elif x==3:
            img=r
            cv.imshow('image',img)
            x=1
            
    elif k == ord('s'):
        cv.createTrackbar('trackbar','image',0,5,slideHandler)
        
    elif k==ord('d'):
        img = cv.pyrDown(img,(c/2, r/2))
        cv.imshow('image',img)
        
    elif k==ord('D'):
        img = cv.pyrDown(img,(c/2, r/2))
        img = cv.filter2D(img,-1)
        cv.imshow('image',img)
        
    elif k == ord('x'):
        img = cv.Sobel(imge,cv.CV_64F,1,0)
        #gx, gy = np.gradient(img)
        img =cv.normalize(img,0,255,cv.NORM_MINMAX)
        cv.imshow('image',img)
        
    elif k == ord('y'):
        img = cv.Sobel(imge,cv.CV_64F,0,1)
        img =cv.normalize(img,0,255,cv.NORM_MINMAX)
        cv.imshow('image',img)
        
    elif k == ord('m'):
        hori = cv.Sobel(imge,cv.CV_64F,1,0)
        h=cv.normalize(hori,0,255,cv.NORM_MINMAX)
        veri = cv.Sobel(imge,cv.CV_64F,1,0)
        v=cv.normalize(veri,0,255,cv.NORM_MINMAX)
        img= np.hypot(h,v)
        cv.imshow('image',img)
    elif k== ord('p'):
        img = cv.imread("/Users/charusaxena/Desktop/bs.png")
        xval = np.arange(img.shape[0])
        yval = np.arange(img.shape[1])
        x,y = np.meshgrid(xval,yval,indexing='ij')


        hori = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
        hor=cv.normalize(hori,0,255,cv.NORM_MINMAX)
        veri = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
        ver=cv.normalize(veri,0,255,cv.NORM_MINMAX)
#first3 = numpy.dstack(firstmatrices)
#x = np.linspace(0,img[0],1)
#y = np.linspace(img[1],0,1)
        u = v = np.zeros((11,11))
        u[0,0]=0.2
        
        plt.quiver(x,y,u,v,scale=1, units ='width')
#qk = plt.quiverkey(q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',coordinates='figure')
#plt.imshow(img,origin='lower')
        plt.show()


        
    elif k == ord('r'):
        cv.createTrackbar('rotate','image', 0,360,rotateHandler)
        
    elif k == ord('h'):
        print ("program description:")
        print ("This program performs simple image manipulation")
        print ("Enter 'assignment2' to run program with camera")
        print ("Enter 'assignment2-Charu Saxena' to run program with custom image input")
        print ("This program takes following command lines")
        print ("'i' -> reload original image")
        print ("'w' -> save current window")
        print ("'g' -> image to grayscale")
        print ("'G' -> image to grayscale without inbuilt function")
        print ("'s' -> To smooth the image according to slider")
        print ("'S' -> To smooth the image according to slider without inbuilt function")
        print ("'d' -> downasample without smoothing")
        print ("'D' -> downasample with smoothing")
        print ("'x' -> Convolution with x derivative")
        print ("'y' -> Convolution with y derivative")
        print ("'m' -> magnitude of gradient")
        print ("'p' -> Plotted gradient with adjustable vector length")
        print ("'r' -> Adjustable image rotation")
        print ("'h' -> Diaplay Help menu")

        cv.putText(img, 'baboon ',(130,130),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv.LINE_AA)
        cv.imshow('image',img)
        
cv.destroyAllWindows()




