
"""Utilities to deal with the cameras of human3.6m"""

from __future__ import division

import h5py
import numpy as np
from helper import timeit
import torch


class Camera_HM():
    def __init__(self,sub,index,R, T, f, c, k, p,name):
        self. R = R
        self.R_cuda = torch.from_numpy(self.R).float().cuda()
        self. T = T
        self.T_cuda = torch.from_numpy(self.T).float().cuda()
        self. f = f
        self. c = c
        self. k = k
        self. p = p
        self.sub = sub
        self.index = index
        self.name = name

        self.rt = np.zeros((4, 4))
        self.rt[:3, :3] = self.R
        self.rt[3, 3] = 1
        self.rt[:3, 2:3] = self.T
        self.rt_ts = torch.from_numpy(self.rt).float()
        self.rt_cuda = torch.from_numpy(self.rt).float().cuda()
        self.fc = np.zeros((3, 4))
        self.fc[0, 0] = self.f[0]
        self.fc[1, 1] = self.f[1]
        self.fc[0, 3] = self.c[0]
        self.fc[1, 3] = self.c[1]
        self.fc[2, 2] = 1
        self.fc_ts = torch.from_numpy(self.fc).float()
        self.fc_cuda = torch.from_numpy(self.fc).float().cuda()

        self.f_cuda = torch.from_numpy(self.f).float().cuda()
        self.c_cuda = torch.from_numpy(self.c).float().cuda()

    def world2pixDist(self, P):
      """
      Project points from 3d to 2d using camera parameters
      including radial and tangential distortion

      Args
        P: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
      Returns
        Proj: Nx2 points in pixel space
        D: 1xN depth of each point in camera space
        radial: 1xN radial distortion per point
        tan: 1xN tangential distortion per point
        r2: 1xN squared radius of the projected points before distortion
      """

      # P is a matrix of 3-dimensional points
      assert len(P.shape) == 2
      assert P.shape[1] == 3
      timeit('1')

      N = P.shape[0]
      X = self.R.dot( P.T - self.T ) # rotate and translate
      XX = X[:2,:] / X[2,:]
      r2 = XX[0,:]**2 + XX[1,:]**2
      radial = 1 + np.einsum( 'ij,ij->j', np.tile(self.k,(1, N)), np.array([r2, r2**2, r2**3]) )
      tan = self.p[0]*XX[1,:] + self.p[1]*XX[0,:]

      XXX = XX * np.tile(radial+tan,(2,1)) + np.outer(np.array([self.p[1], self.p[0]]).reshape(-1), r2 )

      Proj = (self.f * XXX) + self.c
      Proj = Proj.T.astype(np.int32)

      D = X[2,]

      # return Proj, D, radial, tan, r2
      return Proj

    def world2pix(self, p):
        """
        Project points from 3d to 2d using camera parameters
        including radial and tangential distortion

        Args
          P: Nx3 points in world coordinates
          R: 3x3 Camera rotation matrix
          T: 3x1 Camera translation parameters
          f: (scalar) Camera focal length
          c: 2x1 Camera center
          k: 3x1 Camera radial distortion coefficients
          p: 2x1 Camera tangential distortion coefficients
        Returns
          Proj: Nx2 points in pixel space
          D: 1xN depth of each point in camera space
          radial: 1xN radial distortion per point
          tan: 1xN tangential distortion per point
          r2: 1xN squared radius of the projected points before distortion
        """

        # P is a matrix of 3-dimensional points
        assert len(p.shape) == 2
        assert p.shape[1] == 3
        if type(p) is np.ndarray:
            p = self.R.dot(p.T - self.T)
            u = (p[0] / p[2] * self.f[0] + self.c[0]).astype(int)
            v = (p[1] / p[2] * self.f[1] + self.c[1]).astype(int)
            return np.array((u, v)).T.astype(np.int32)
        elif type(p) is torch.Tensor and p.is_cuda:
            p = self.R_cuda.matmul(p.t()-self.T_cuda)
            u = (p[0] / p[2] * self.f_cuda[0] + self.c_cuda[0]).unsqueeze(1)
            v = (p[1] / p[2] * self.f_cuda[1] + self.c_cuda[1]).unsqueeze(1)
            return torch.cat((u,v),1).long()
            # p = self.fc_cuda.matmul(p)
            # p[0, :] = p[0, :] / p[2, :]
            # p[1, :] = p[1, :] / p[2, :]
            # return p[:2, :].t().long()
        else:
            p = self.rt_ts.matmul(p.t())
            p = self.fc_ts.matmul(p)
            p[0, :] = p[0, :] / p[2, :]
            p[1, :] = p[1, :] / p[2, :]
            return p[:2, :].t().long()

    def world_to_camera_frame(self,P):
      """
      Convert points from world to camera coordinates

      Args
        P: Nx3 3d points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
      Returns
        X_cam: Nx3 3d points in camera coordinates
      """

      assert len(P.shape) == 2
      assert P.shape[1] == 3

      X_cam = self.R.dot( P.T - self.T ) # rotate and translate

      return X_cam.T

    def camera_to_world_frame(self,P):
      """Inverse of world_to_camera_frame

      Args
        P: Nx3 points in camera coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
      Returns
        X_cam: Nx3 points in world coordinates
      """

      assert len(P.shape) == 2
      assert P.shape[1] == 3

      X_cam = self.R.T.dot( P.T ) + self.T # rotate and translate

      return X_cam.T


def load_camera_params( hf, path ):
  """Load h36m camera parameters

  Args
    hf: hdf5 open file with h36m cameras data
    path: path or key inside hf to the camera we are interested in
  Returns
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    name: String with camera id
  """

  R = hf[ path.format('R') ][:]
  R = R.T

  T = hf[ path.format('T') ][:]
  f = hf[ path.format('f') ][:]
  c = hf[ path.format('c') ][:]
  k = hf[ path.format('k') ][:]
  p = hf[ path.format('p') ][:]

  name = hf[ path.format('Name') ][:]
  name = "".join( [chr(item) for item in name] )

  return R, T, f, c, k, p, name


def load_cameras( bpath, subjects=[1,5,6,7,8,9,11] ):
  """Loads the cameras of h36m

  Args
    bpath: path to hdf5 file with h36m camera data
    subjects: List of ints representing the subject IDs for which cameras are requested
  Returns
    rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
  """
  rcams = []
  cam_dic = {}
  with h5py.File(bpath,'r') as hf:
    for s in subjects:
      for c in range(4): # There are 4 cameras in human3.6m
        cam = load_camera_params(hf, 'subject%d/camera%d/{0}' % (s,c+1))
        rcams.append(Camera_HM('S'+str(s),c,*cam))
      cam_dic['S' + str(s)] = rcams[-4:]

  return cam_dic
