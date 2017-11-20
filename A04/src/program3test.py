#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 11:38:20 2017

@author: charusaxena
"""

import numpy as np
import math
import sys

with open(sys.argv[1], 'r') as f:
    content = f.readlines()
content = [x.strip() for x in content] 
data = content


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

mat,worldpoint,imagepoint = make_A_matrix(data)
projection_matrix=projection(mat)

for i in range(len(data)):
    ui,vi,wi=np.matmul(projection_matrix,worldpoint[0])
    u=ui/wi
    v=vi/wi
    print("\npoint are :",u,v)