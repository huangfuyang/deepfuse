import numpy as np
from dataset.camera_hm import *
from server_setting import *
import os
from params import *
from visualization import *
p = np.load('/home/hfy/data/human36/S1/Videos/Directions/00005.npy')
mcv = p[0]
label = p[1]
mid = p[2]
leng = p[3]
cams = load_cameras(os.path.join(HM_PATH,'cameras.h5'))
c = cams['S1']
base = mid - np.repeat(leng,3) * (NUM_VOXEL / 2 - 0.5)
# leng = np.repeat(leng,JOINT_LEN * 3)
# base = np.repeat(mid.reshape,JOINT_LEN)

label = label.reshape(-1,3)
x,y,z = np.where(mcv[0]!=0)
b = mcv[0][mcv[0]!=0]
g = mcv[1][mcv[0]!=0]
r = mcv[2][mcv[0]!=0]
x = x * leng + base[0]
y = y * leng + base[1]
z = z * leng + base[2]
d = np.array((x,y,z)).T
res = c[0].world2pix(d)
img = np.zeros(shape=(1002,1002,3),dtype=np.ubyte)
img[res[:,1],res[:,0],0] = r
img[res[:,1],res[:,0],1] = g
img[res[:,1],res[:,0],2] = b
res = c[0].world2pix(label)
drawcirclecv(img,res)
import cv2
cv2.imshow('',img)
cv2.waitKey(0)
plot_voxel(mcv[0])
plot_voxel(mcv[1])
plot_voxel(mcv[2])

