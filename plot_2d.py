import numpy as np
import torch
import os
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io
from dataset.human36rgb import *
import pylab
from server_setting import *
from dataset import camera_hm

skeleton = [[15,14],[14,13],[13,12],[12,0],[0,6],[0,1],[6,7],[7,8],[1,2],[2,3],[17,18],[18,19],[13,17],[13,25],[25,26],[26,27]]

'''skeleton_colors = [(np.array([100, 64,  183]) /255.).tolist(),
                   (np.array([57,  85,  179]) /255.).tolist(),
                   (np.array([13,  152, 241]) /255.).tolist(),
                   (np.array([252, 88,  40])  /255.).tolist(),
                   (np.array([141, 194, 81])  /255.).tolist(),
                   (np.array([207, 217, 69])  /255.).tolist(),
                   (np.array([7,   170, 241]) /255.).tolist(),
                   (np.array([244, 68,  56])  /255.).tolist(),
                   (np.array([154, 48,  174]) /255.).tolist(),
                   (np.array([250, 234, 73])  /255.).tolist(),
                   (np.array([252, 152, 49])  /255.).tolist(),
                   (np.array([122, 85,  74])  /255.).tolist(),
                   (np.array([158, 158, 158]) /255.).tolist(),
                   (np.array([97,  125, 140]) /255.).tolist(),
                   (np.array([254, 192, 58])  /255.).tolist(),
                   (np.array([25,  150, 136]) /255.).tolist()]
'''
path = '/home/alzeng/remote/data/human36/S9/MySegmentsMat/ground_truth_bs/Waiting 1/00768.npz'
path_test = '/home/alzeng/remote/fyhuang/alzeng/deepfuse/result/human_p2.npz'
npz = np.load(path_test)
#npz = np.load(path_test)
label = npz['label']

result = npz['result']
label = result[1050]

label = label.reshape(-1,3)
cams = camera_hm.load_cameras(os.path.join(HM_PATH,'cameras.h5'))
cams = cams['S9']
gt = cams[1].world2pix(torch.from_numpy(label).float().cuda())

gt = gt.cpu()
'''
x = np.array(gt[:,0]+40)
y = np.array(gt[:,1]+20)
'''
x = np.array(gt[:,0])
y = np.array(gt[:,1])
fig = plt.figure()

# load image
#image_path = '/home/alzeng/remote/fyhuang/data/NewHuman3.6/processed/S9/Waiting-2/imageSequence/55011271/img_000769.jpg'
image_path = '/home/alzeng/remote/data/human36/S9/MySegmentsMat/ground_truth_bs/img/00768.jpg'
I = io.imread(image_path)
plt.imshow(I);
plt.axis('off')
ax = plt.gca()
ax.set_autoscale_on(False)
sks = np.array(skeleton)
for cidx, sk in enumerate(sks):
    plt.plot(x[sk], y[sk], linewidth=2, color='grey')

a = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
b = [2,3,7,8,18,19,26,27]
for i in a:
    if i in b:
        plt.plot(x[i],y[i],'o',markersize=5, markerfacecolor='r', markeredgecolor='r', markeredgewidth=2)
    else:
        plt.plot(x[i], y[i], 'o',markersize=5, markerfacecolor='b',markeredgecolor='b', markeredgewidth=2)

plt.show()