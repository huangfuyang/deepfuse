import numpy as np
from params import *
import torch
from time import time

is_init = False
def init():
    global coords,start,end,size,is_init
    size = NUM_VOXEL
    start = -(NUM_VOXEL / 2 - 0.5)
    end = NUM_VOXEL / 2 - 0.5
    coord = np.arange(start=start, stop=end + 0.1, step=1.0, dtype=np.float32)
    x = np.repeat(coord, size * size)
    y = np.tile(np.repeat(coord, size), size)
    z = np.tile(coord, size * size)
    coords = (np.column_stack((x, y, z)))
    coords = torch.from_numpy(coords).cuda()
    is_init = True


def data_augmentation3D(pvh,label,mid,leng):
    if not is_init:
        init()
    r = np.radians(np.random.rand() * ROTATE_ANGLE * 2 - ROTATE_ANGLE)
    # for gt
    rotation = np.array([[np.math.cos(r), 0, np.math.sin(r)],
                         [0, 1, 0],
                         [-np.math.sin(r), 0, np.math.cos(r)]], dtype=np.float32)
    # for matrix
    rotation_reverse = np.array([[np.math.cos(-r), 0, np.math.sin(-r)],
                                 [0, 1, 0],
                                 [-np.math.sin(-r), 0, np.math.cos(-r)]], dtype=np.float32)

    # leng = leng * (NUM_VOXEL / NUM_GT_SIZE)
    # r_leng = leng.repeat(JOINT_LEN * 3)
    r_mid = np.tile(mid, JOINT_LEN)
    label = (label - r_mid).reshape(-1, 3)
    label = label.dot(rotation) / leng
    max_p = np.max(label)
    min_p = np.min(label)
    max_p = max(abs(max_p), abs(min_p))
    scale = 1
    if max_p > end:
        scale = end / max_p
    rotation_reverse = torch.from_numpy(rotation_reverse).cuda()

    label = (label * leng * scale).reshape(-1) + r_mid
    n_coords = (coords.matmul(rotation_reverse) / scale + size / 2).long()
    n_coords[n_coords < 0] = 0
    n_coords[n_coords > (NUM_VOXEL - 1)] = NUM_VOXEL - 1
    if pvh.dim()==3:
        pvh = pvh[n_coords[:, 0],n_coords[:, 1],n_coords[:, 2]].reshape(size, size, size)
    else:
        for i in range(pvh.shape[0]):
            pvh[i] = (pvh[i][n_coords[:, 0],n_coords[:, 1],n_coords[:, 2]]).reshape(size, size, size)

    return pvh,label,mid,leng


def random_shut(data, ratio = RANDOM_RATIO):
    """
    randomly shut down a camera from multi-view video
    :param data: multi-view video in numpy array
    :param ratio: random ratio
    :return: result
    """
    if np.random.rand()>ratio:
        index = int(np.random.rand()*NUM_CAM)
        data[index] = 0
    return data