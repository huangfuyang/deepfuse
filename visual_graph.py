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

def drawcircle(img,pc,r = 3):
    pc1 = pc.copy()+r
    pc = pc-r
    pc = np.concatenate((pc,pc1),axis=1)
    draw = ImageDraw.Draw(img)
    for i in pc:
        print(i)
        draw.ellipse(tuple(i),fill=128)
    return img


def drawcirclecv(img, p,r=8):
    p = totuple(p)
    # ALzeng please change node sequence according to human3.6 definition
    cv2.line(img, p[15], p[14], (128,128,128),thickness=7)
    cv2.line(img, p[13], p[14], (128,128,128),thickness=7)
    cv2.line(img, p[13], p[12], (128,128,128),thickness=7)
    cv2.line(img, p[0], p[12], (128,128,128),thickness=7)
    cv2.line(img, p[0], p[6], (128,128,128),thickness=7)
    cv2.line(img, p[1], p[0], (128,128,128),thickness=7)
    cv2.line(img, p[6], p[7], (128,128,128),thickness=7)
    cv2.line(img, p[8], p[7], (128,128,128),thickness=7)
    cv2.line(img, p[1], p[2], (128,128,128),thickness=7)
    cv2.line(img, p[3], p[2], (128,128,128),thickness=7)
    cv2.line(img, p[17], p[18], (128,128,128),thickness=7)
    cv2.line(img, p[19], p[18], (128,128,128),thickness=7)
    cv2.line(img, p[13], p[17], (128,128,128),thickness=7)
    cv2.line(img, p[13], p[25], (128,128,128),thickness=7)
    cv2.line(img, p[25], p[26], (128,128,128),thickness=7)
    cv2.line(img, p[27], p[26], (128,128,128),thickness=7)

    for i in p:
        # cv2.circle(img,i,r,(0, 69 ,255),thickness=-1)
        cv2.circle(img,i,r,(255, 0 ,0),thickness=-1)

    for i in [2,3,7,8,18,19,26,27]:
        cv2.circle(img,p[i],r,(0, 69 ,255),thickness=-1)


def plot_tsdf(data, max_p, mid_p, label=None, axis=0):
    data = np.squeeze(data[axis,:,:,:]).reshape(-1)
    scale = np.zeros(data.shape)+0.5
    scale[data == 1] = 0
    scale[data == -1] = 0
    scale[data == 0] = 0
    data = (data+1)/2
    graph = points3d(np.tile(np.arange(-VOXEL_RES / 2, VOXEL_RES / 2), (VOXEL_RES * VOXEL_RES)),
             np.tile(np.repeat(np.arange(-VOXEL_RES / 2, VOXEL_RES / 2), VOXEL_RES), VOXEL_RES),
             np.repeat(np.arange(-VOXEL_RES / 2, VOXEL_RES / 2), (VOXEL_RES * VOXEL_RES)),
                     scale_factor=1)
    graph.glyph.scale_mode = 'scale_by_vector'

    if label is not None:
        label = (label.reshape(-1, 3) - mid_p).reshape(-1) / max_p + 0.5
        label[label < 0] = 0
        label[label >= 1] = 1
        label = (label * VOXEL_RES).astype(int)
        label = label.reshape(-1,3)
        label = label[:,0]+label[:,1]*VOXEL_RES+label[:,2]*VOXEL_RES*VOXEL_RES
        scale[label] = 1.5
    graph.mlab_source.dataset.point_data.vectors = np.repeat(scale, 3).reshape(-1,3)
    graph.mlab_source.dataset.point_data.scalars = data
    show()


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
    print(gt.shape)
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


def draw_pvh(data,label=None):
    if label is None:
        plot_voxel(data)
    else:
        plot_voxel_label(data,label)



def plot_voxel_label_both(data):
    x,y,z = np.where(data==1)
    x1,y1,z1 = np.where(data==2)
    scale = np.ones(x.shape[0]+x1.shape[0])
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


def plot_voxel_label_2(data, label, min_p, voxel_length):
    if label is not None:
        label = label.reshape(-1,3).copy()
        label = label-np.repeat(np.expand_dims(min_p,axis=0),label.shape[0],axis=0)
        label = (label/voxel_length).astype(int)
        data = data.copy()
        data[label[:,0],label[:,1],label[:,2]] = 2
    x,y,z = np.where(data==1)
    x1,y1,z1 = np.where(data==2)
    scale = np.ones(x.shape[0]+x1.shape[0])
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

def plot_voxel_gt(data, gt, min_p, voxel_length):
    if gt is not None:
        gt = gt.reshape(-1, 3).copy()
        gt = gt - np.repeat(np.expand_dims(min_p, axis=0), gt.shape[0], axis=0)
        gt = (gt / voxel_length).astype(int)
        data = data.copy()
        data[gt[:, 0], gt[:, 1], gt[:, 2]] = 2
    x,y,z = np.where(data==1)
    x1,y1,z1 = np.where(data==2)
    scale = np.ones(x.shape[0]+x1.shape[0])
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


def plot_voxel_result(data, label, mid_p, voxel_length):
    label = label.reshape(-1,3)-np.repeat(np.expand_dims(mid_p,axis=0),label.shape[0],axis=0)
    label = (label/voxel_length).astype(int)
    data[label[:,0],label[:,1],label[:,2]] = 2
    x,y,z = np.where(data==1)
    x1,y1,z1 = np.where(label==2)
    scale = np.ones(x.shape[0]+x1.shape[0])
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


def plot_pointcloud(data, label = None):
    scale = np.ones(data.shape[0])/4
    if label is not None:
        label = label.reshape(-1,3)
        data = np.concatenate((data,label),axis=0)
        scale = np.concatenate((scale,np.ones(label.shape[0])),axis=0)
    graph = points3d(data[:,0],data[:,1],data[:,2],scale,scale_factor = 3)
    # graph.glyph.scale_mode = 'scale_by_vector'
    #
    # graph.mlab_source.dataset.point_data.vectors = np.repeat(scale, 3).reshape(-1, 3)
    # graph.mlab_source.dataset.point_data.scalars = scale
    show()

if __name__ == '__main__':
    # 1 channel image
    img_matte = np.zeros((1000,1000,1))
    # change 1 channel to 3 channel images
    img_rgb = np.tile(img_matte.copy(), (1, 1, 3))
    # r is  N points to draw   shape = N * 2
    r = (np.random.rand(17,2)*1000).astype(np.int32)
    #path = '/home/alzeng/remote/fyhuang/data/NewHuman3.6/extracted/S5/Videos/WalkDog 1/00000.npz'
    path = '/home/alzeng/remote/data/human36/S9/MySegmentsMat/ground_truth_bs/Greeting/00192.npz'
    npz = np.load(path)
    label = npz['label']
    label = np.reshape(label, (-1, 3))
    x = label[:, 0]
    y = label[:, 1]
    drawcirclecv(img_rgb, r)
    cv2.imshow('',img_rgb)
    cv2.waitKey(0)
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
