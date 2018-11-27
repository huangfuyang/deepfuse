from mayavi.mlab import *
import numpy as np
from params import *
import PIL.ImageDraw as ImageDraw
import cv2


def visualize3d(pc):
    graph = points3d(pc[:, 0], pc[:, 1], pc[:, 2],scale_factor=1)
    show()


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def drawcirclecv(img, p,r=3):
    for i in p:
        cv2.circle(img,totuple(i),r,(128,128,128),thickness=-1)


def plot_voxel(data):
    x,y,z = np.where(data==1)
    scale = np.ones(x.shape[0])
    graph = points3d(x,y,z)
    show()


def plot_voxel_label(data, label):
    x,y,z = np.where(data>0)
    label = label.reshape(-1,3)
    x1,y1,z1 = label[:,0],label[:,1],label[:,2]
    scale = np.ones(x.shape[0]+x1.shape[0])*0.5
    scale[x.shape[0]:] = 2
    graph = points3d(np.append(x,x1),np.append(y,y1),np.append(z,z1),
                     scale_factor=1)
    graph.glyph.scale_mode = 'scale_by_vector'

    # if label is not None:
    #     label = (label.reshape(-1, 3) - mid_p).reshape(-1) / max_p + 0.5
    #     label[label < 0] = 0
    #     label[label >= 1] = 1
    #     label = (label * size).astype(int)
    #     label = label.reshape(-1, 3)
    #     label = label[:, 0] + label[:, 1] * size + label[:, 2] * size * size
    #     scale[label] = 1.5
    graph.mlab_source.dataset.point_data.vectors = np.repeat(scale, 3).reshape(-1, 3)
    graph.mlab_source.dataset.point_data.scalars = scale-1
    show()


def plot_gt(gt):
    joint = gt.shape[0]
    scalar, scale, x, y, z = np.empty(0),np.empty(0),np.empty(0,dtype=np.int32),np.empty(0,dtype=np.int32),np.empty(0,dtype=np.int32)
    print gt.shape
    for i in range(joint):
        x1, y1, z1 = np.where(gt[i] >0)
        x = np.append(x,x1)
        y = np.append(y,y1)
        z = np.append(z,z1)
        scalar = np.append(scalar, np.zeros(x1.shape[0])+float(i)/JOINT_LEN)
        scale = np.append(scale, gt[i][x1,y1,z1])
    graph = points3d(x,y,z,
                     scale_factor=1)
    graph.glyph.scale_mode = 'scale_by_vector'

    graph.mlab_source.dataset.point_data.vectors = np.repeat(scale, 3).reshape(-1, 3)
    graph.mlab_source.dataset.point_data.scalars = scalar
    show()


def draw_mvc(data,label=None):
    if label is None:
        plot_voxel(data)
    else:
        plot_voxel_label(data,label)



if __name__ == '__main__':
    pass
    # p = np.array([  0.964 , 33.281 ,  5.105])
    # p = np.array([0,0,0]).reshape(1,3)
    # cams = init_cameras(TC_PATH)
    # pc = np.array([0,0,0])
    # for cam in cams:
    #     pt = cam.world2cam(p).T
    #     pc = np.vstack((pc,cam.cam_center()[0]))
    # print pc
    # graph = points3d(pc[0,:],pc[1,:],pc[2,:])
    # show()
