#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:01:16 2017

@author: charusaxena
"""

import random
import numpy as np
import math
import sys

with open(sys.argv[1], 'r') as f:
    content = f.readlines()
content = [x.strip() for x in content] 
data = content

with open(sys.argv[2], 'r') as q:
    ransac_params = q.readlines()
ransac_params = [x.strip() for x in ransac_params] 
max_k = float(ransac_params[0])
n=int(ransac_params[1])
p=float(ransac_params[2])

def make_A_matrix(data):
    data =np.array(data)
    total_matrix = []
    world_point=[]
    image_point=[]
    for i in data:
        X,Y,Z,xi,yi = i.split(' ')
        X = float(X)
        Y = float(Y)
        Z = float(Z)
        xi = float(xi)
        yi = float(yi)
        a= X,Y,Z,1,0,0,0,0,-xi*X,-xi*Y,-xi*Z,-xi
        b = 0,0,0,0, X,Y,Z,1,-yi*X,-yi*Y,-yi*Z,-yi
        c= X,Y,Z,1
        
        
        world_point.append(c)
        d= xi,yi
        image_point.append(d)
        matrix= []
        matrix.extend(a)
        total_matrix.append(matrix)
        matrix=[]
        matrix.extend(b)
        total_matrix.append(matrix)
    return total_matrix,world_point,image_point

def projection(matrix):
    U, s, V = np.linalg.svd(matrix)
    m = V.T[:,-1].reshape(3,4)
    return m

def thres(projection_matrix,test):
    a,b,c = make_A_matrix(test)
    distance=[]
    for i in range(len(test)):
        
        ui,vi,wi=np.matmul(projection_matrix,b[i])
        u=ui/wi
        v=vi/wi
        d,e = c[i]
        dist = math.sqrt(math.pow(d-u,2)+ math.pow(e-v,2))
        distance.append(dist)
    
    return (1.5*(np.median(distance)))


def inliers(projectionMatrix,data,thres):
    a,world_point,image_point = make_A_matrix(data)
    
    myinliers=[]
    for i in range(len(data)):
        
        ui,vi,wi=np.matmul(projection_matrix,world_point[i])
        u=ui/wi
        v=vi/wi
        a,b = image_point[i]
        dist = math.sqrt(math.pow(a-u,2)+ math.pow(b-v,2))
        if dist < thres:
            myinliers.append(data[i])
    
    return myinliers,len(myinliers)

def print_parameters(projection_matrix):
    K_star_R_star = projection_matrix[:3,:3]
    K_star_T_star = projection_matrix[:,-1]
    print("Intrinsic and extrinsic parameteres are :\n")
    Ro = 1/np.linalg.norm(K_star_R_star[-1])  
    Uo = ((Ro)**2)*np.dot(K_star_R_star[:1],K_star_R_star[-1])
    Vo = ((Ro)**2)*np.dot(K_star_R_star[1:2],K_star_R_star[-1])
    alphav =math.sqrt( (((Ro)**2)*np.dot(K_star_R_star[1:2],K_star_R_star[1:2].T)) -((Vo)**2)) 
    s =(alphav * np.dot(np.cross(K_star_R_star[:1],K_star_R_star[-1]),np.cross(K_star_R_star[1:2],K_star_R_star[-1]).T))
    #if (s<0):
        #s= 0
    alphau = math.sqrt((math.pow(Ro,2)*np.dot(K_star_R_star[:1],K_star_R_star[:1].T)) - math.pow(s,2) - math.pow(Uo,2))
    K_star = np.array([[alphau,s,Uo],[0,alphav,Vo],[0,0,1]]) 
    print("\nthe sign of epsilon",np.sign(K_star_T_star[-1]))
    T_star = np.sign(K_star_T_star[-1])*Ro*(np.matmul((np.linalg.inv(K_star)),K_star_T_star))
    R3 = np.sign(K_star_T_star[-1])*Ro*K_star_R_star[-1]
    R1= math.pow(Ro,2)/alphav*(np.cross(K_star_R_star[1:2],K_star_R_star[-1]))
    R2 = np.cross(R3,R1)
    
    R_matrix = np.array([[R1],[R2],[R3]])
    print("\n Ro value is:",Ro)
    print("\n Uo value is",Uo)
    print("\nthe value Vo",Vo)
    print("\nthe value alphav",alphav)
    print("\nthe value S",s)
    print("\nthe value of alphau",alphau)
    print(K_star)
    print("\nvalue of T_star",T_star) 
    print("\n",R_matrix)

#Ransac Implementation

count=0
all_S=[]
all_len=[]
k=math.ceil(np.log(1-p)/np.log(1-math.pow(0.5,n)))
#ransac
while count<k:
    count=count+1
    if (count>max_k):
        break
    else:
        test = random.sample(content,n)
        mat,worldpoint,imagepoint = make_A_matrix(test)
        
        projection_matrix=projection(mat)
        threshold  = thres(projection_matrix,test)
        
        new_s,length = inliers(projection_matrix,data,threshold)
       
        w = len(new_s)/len(content)
        k=math.ceil(np.log(1-p)/np.log(1-math.pow(w,n)))
        
        if length > 6:
            all_S.append(new_s)
            all_len.append(length)
            
b = all_S[np.argmax(all_len)]
matt,worldpoint,imagepoint = make_A_matrix(all_S[np.argmax(all_len)])
final_projection_matrix=projection(matt)        
print_parameters(final_projection_matrix)

##mean square error
g,world,image = make_A_matrix(data)
estimates_image_point=[]
for i in range(len(data)):
        uei,vei,wei=np.matmul(final_projection_matrix,world[i])
        ue=uei/wei
        ve=vei/wei
        de,ee = image[i]
        error = ((math.pow(de-ue,2)+ math.pow(ee-ve,2)))/len(data)

print("\nmean squared error is:",error)
##3


