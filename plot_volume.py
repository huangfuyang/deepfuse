# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#data = np.random.randint(0, 255, size=[40, 40, 40])
#path = '/home/alzeng/remote/data/human36/S1/MySegmentsMat/ground_truth_bs/Directions/00026.npz'
path = '/home/alzeng/remote/fyhuang/data/NewHuman3.6/extracted/S5/Videos/WalkDog 1/00000.npz'
npz = np.load(path)

label = npz['label'] #96dimension for gt kpt
pvh = npz['mcv'] #4camera volume index with shape [4,64,64,64]
pvh = pvh.astype(np.uint8)
for j in pvh:
    print type(j)
#pvh = torch.from_numpy(pvh).cuda().float()
#label = np.reshape(label,(-1,3))
    x,y,z = np.where(pvh>=1) #pvh[i]==1 means just look this ith camera,i=0,1,2,3
    v = pvh[pvh>=1]
    print('x',x,'y',y,'z',z,'v',v)
    # x = label[:,0]
    # y = label[:,1]
    # z = label[:,2]

    #   ax = plt.subplot(111, projection='3d')
    fig = plt.figure()
    ax = Axes3D(fig)
    #ax.scatter(x,y,z,c='y')
    ax.scatter(x,y,z,c='r')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    plt.show()