#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:38:35 2017

@author: charusaxena
"""

import cv2
import numpy as np
from numpy import mean,std
import  sys

infile = sys.argv[1]
left = cv2.imread(infile)

infine2  = sys.argv[2]
right = cv2.imread(infine2)
#left = cv2.imread("/Users/charusaxena/Desktop/corridor-l.tiff")
#right = cv2.imread("/Users/charusaxena/Desktop/corridor-r.tiff")

#left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
#right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

rows_r,cols_r,_ = right.shape
rows_l,cols_l,_ = left.shape
   
  

#normlaise the list of points we enter and return the M matrix
def normalise(pts1):
    
    a,b,z =std(pts1, axis=0)
    sig=np.zeros((3,3))
    for i in range(0,3):
        for j in range(0,3):
            if i==j==0:
                sig[i,j] = 1/a
            elif i==j==1:
                sig[i,j] = 1/b
            elif i==j==2:
                sig[i,j]=1
    c,d,q=mean(pts1, axis=0)
    mu = np.zeros((3,3))
    for i in range(0,3):
        for j in range(0,3):
            if i==j==0:
                mu[i,j] = 1      
            elif i==j==1:
                mu[i,j] = 1
            elif i==j==2:
                mu[i,j]=1 
    mu[0,2]=-c
    mu[1,2]=-d
    M =np.matmul(sig,mu)
    
    return M             

# make a f matrix , but here used for making fundamental matrix for normalised 
#points and make sure that the rank is 2.
           
def make_Fprime_matrix(left,right):

    total_matrix = []
    for i,j in zip(left,right):
        lx,ly,d = i
        rx,ry,e= j
        a= lx*rx,ly*rx,rx,lx*ry,ly*ry,ry,lx,ly,1
        matrix= []
        matrix.extend(a)
        total_matrix.append(matrix)

    U, s, V = np.linalg.svd(total_matrix)

    m = V.T[:,-1].reshape(3,3)
    U, s, V = np.linalg.svd(m)
#making it rank 2
    S = np.diag(s)

    row=S.shape[0]-1
    col=S.shape[1]-1
    S[row,col]=0
    #print("\nprint S",S)
#returnin the f prime.
    return (np.dot(U,np.dot(S,V)))



## this function will allow to enter points using mouse left key and get x,y coordinates
def on_mouse(event, x, y, flags, params):
   global ix,iy,drawing   
   
   if event==cv2.EVENT_LBUTTONDOWN:
       
       ix,iy=x,y
       
       #print("points getting")

a='a'

##this function is used to allow points for making epipolar points
def press_mouse(event, x, y, flags, params):
   global ix,iy,drawing   
   
   if event==cv2.EVENT_LBUTTONDOWN:
       #acv2.putText(left,str(a),(ix,iy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
       ix,iy=x,y
       
count=0
xy_left=[]
total_left=[]
total_right=[]

while(1):
   cv2.imshow('left',left)
   cv2.setMouseCallback('left',on_mouse, 0)
      
   
   k = cv2.waitKey(20) & 0xFF
   if k == 27:
       break
   elif k == ord('a'):
       count = count +1
       cv2.putText(left,str(count),(ix,iy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
       #print (ix,iy)
       xy_left = []
       
       xy_left.append(ix)
       xy_left.append(iy)
       xy_left.append(1)
       total_left.append(xy_left)
       
       #print("\nadded to list",total_left)
       

#print("\nfinal list ",total_left)



counter = 0
while(1):
   cv2.imshow('right',right)
   cv2.setMouseCallback('right',on_mouse, 0)

   
   k = cv2.waitKey(20) & 0xFF
   if k == 27:
       break
   elif k == ord('b'):
       counter = counter +1
       cv2.putText(right,str(counter),(ix,iy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
       #print (ix,iy)
       xy_right = []
       
       xy_right.append(ix)
       xy_right.append(iy)
       xy_right.append(1)
       total_right.append(xy_right)
       #print("\nadded to list right",total_right)




#print("\nnormal F for test not final ",make_Fprime_matrix(total_left,total_left))
norm_left=[]
norm_right=[]

M_left = normalise(total_left)
for i in total_left:
    new= np.matmul(M_left,i)
    norm_left.append(new)


M_right = normalise(total_right)
for i in total_right:
    new2 =np.matmul(M_right,i)
    norm_right.append(new2)


#print("\n\nfinallyy normlaised left ",norm_left)
#print("\nfinallyy  normalised right ",norm_right)

f_prime =make_Fprime_matrix(norm_left,norm_right)
#print("\n prinintg f prime: ",f_prime)


f=np.dot(M_left.T,np.dot(f_prime,M_right))
print("\nfinal fundamental matrix :\n ",f)

def epipolar(f_matrix):
   # f_matrix=f_matrix.T 
    U, s, V = np.linalg.svd(f_matrix)
    #print(" v \n",V)
    left_epipolar = V.T[:,-1]
    right_epipolar = U[:,-1]
    return left_epipolar,right_epipolar

el,er=epipolar(f)
t,y,u =el
left_epipole = [t/u,y/u]
i,o,p = er
right_epipole = [i/p,o/p]

pts1 = np.int32(total_left)
pts2 = np.int32( total_right)

print("left epipole:\n",left_epipole)
print("right epipole\n:",right_epipole)



while(1):
            cv2.imshow('left',left)
            cv2.setMouseCallback('left',press_mouse, 0)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                break
            elif k == ord('a'):
                right[int(right_epipole[1]),int(right_epipole[0])] = [0,255,255]
                line = np.array([ix,iy,1])
                d = np.matmul(line,f.T)
                a = d[0]
                b = d[1]
                c = d[2]
                if a >= b:
                 
                    for x in range(0,rows_r):
                        y = int((-c -a*x)/b)
                        if y < right.shape[1] and y > 0:
                            right[y,x] = [0,0,255] 
    
                else:
                    #print("else")
                    for y in range(0,cols_r):
                        x = int((-c -b*y)/a)

                        if x < right.shape[0] and x > 0:
                            right[y,x] = [0,255,0]
                cv2.imshow('right',right)
                
while(1):
            cv2.imshow('right',right)
            cv2.setMouseCallback('right',press_mouse, 0)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                break
            elif k == ord('a'):
                #print (ix,iy)
                left[int(left_epipole[1]),int(left_epipole[0])] = [0,255,255]
                line = np.array([ix,iy,1])
                d = np.matmul(line,f.T)
                a = d[0]
                b = d[1]
                c = d[2]
                if a >= b:

                    for x in range(0,rows_r):
                        y = int((-c -a*x)/b)
                        if y < left.shape[1] and y > 0:
                            left[y,x] = [0,0,255] 
    
                else:
                    #print("else")
                    for y in range(0,cols_r):
                        x = int((-c -b*y)/a)
                        if x < left.shape[0] and x > 0:
                            left[y,x] = [0,255,0]
                cv2.imshow('left',left)
                
cv2.destroyAllWindows()





