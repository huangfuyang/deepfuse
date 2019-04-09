import torch.nn as nn
import torch
from params import *
from server_setting import *
from model.res3d import Res3D
from uplayer import Upsample
from softargmax import Softargmax3D


class Hourglass3D(nn.Module):
    def __init__(self, n, nModules, nFeats):
        super(Hourglass3D, self).__init__()
        self.n = n
        self.nModules = nModules
        self.nFeats = nFeats

        _up1_, _low1_, _low2_, _low3_ = [], [], [], []
        for j in range(self.nModules):
            _up1_.append(Res3D(self.nFeats, self.nFeats))
        self.low1 = nn.MaxPool3d(kernel_size=2, stride=2)
        for j in range(self.nModules):
            _low1_.append(Res3D(self.nFeats, self.nFeats))

        if self.n > 1:
            self.low2 = Hourglass3D(n - 1, self.nModules, self.nFeats)
        else:
            for j in range(self.nModules):
                _low2_.append(Res3D(self.nFeats, self.nFeats))
            self.low2_ = nn.ModuleList(_low2_)

        for j in range(self.nModules):
            _low3_.append(Res3D(self.nFeats, self.nFeats))

        self.up1_ = nn.ModuleList(_up1_)
        self.low1_ = nn.ModuleList(_low1_)
        self.low3_ = nn.ModuleList(_low3_)

        self.up2 = Upsample(scale_factor=2)

    def forward(self, x):
        up1 = x
        for j in range(self.nModules):
            up1 = self.up1_[j](up1)

        low1 = self.low1(x)
        for j in range(self.nModules):
            low1 = self.low1_[j](low1)

        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for j in range(self.nModules):
                low2 = self.low2_[j](low2)

        low3 = low2
        for j in range(self.nModules):
            low3 = self.low3_[j](low3)
        up2 = self.up2(low3)

        return up1 + up2


class FuseNet(nn.Module):
    def __init__(self, nStack, nModules, nFeats, nJoint):
        super(FuseNet, self).__init__()
        self.nStack = nStack
        self.nModules = nModules
        self.nFeats = nFeats
        # self.conv1_ = nn.Conv3d(nCHANNEL, self.nFeats/2, bias=True, kernel_size=3, stride=2, padding=1)
        self.conv1_ = nn.Conv3d(nCHANNEL, self.nFeats, bias=True, kernel_size=3, stride=2, padding=1)
        # self.conv1_ = nn.Conv3d(1, 64, bias = True, kernel_size = 7, stride = 2, padding = 3)
        # self.bn1 = nn.BatchNorm3d(self.nFeats/2)
        self.bn1 = nn.BatchNorm3d(self.nFeats)
        self.relu = nn.ReLU(inplace=True)
        # self.r1 = Res3D(self.nFeats/2, self.nFeats)
        self.r1 = Res3D(self.nFeats, self.nFeats)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.r4 = Res3D(self.nFeats, self.nFeats)
        self.r5 = Res3D(self.nFeats, self.nFeats)
        self.drop = nn.Dropout3d()
        self.softarg = Softargmax3D(NUM_GT_SIZE,BATCH_SIZE,nJoint)
        _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_, _reg_ = [], [], [], [], [], [], []
        for i in range(self.nStack):
            _hourglass.append(Hourglass3D(4, self.nModules, self.nFeats))
            for j in range(self.nModules):
                _Residual.append(Res3D(self.nFeats, self.nFeats))
            lin = nn.Sequential(nn.Conv3d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1),
                                nn.BatchNorm3d(self.nFeats), self.relu)
            _lin_.append(lin)
            if i < self.nStack - 1:
                _tmpOut.append(nn.Conv3d(self.nFeats, nJoint, bias=True, kernel_size=1, stride=1))
                _ll_.append(nn.Conv3d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1))
                _tmpOut_.append(nn.Conv3d(nJoint, self.nFeats, bias=True, kernel_size=1, stride=1))
            else:
                _tmpOut.append(nn.Conv3d(self.nFeats, nJoint, bias=True, kernel_size=1, stride=1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin_ = nn.ModuleList(_lin_)
        self.tmpOut = nn.ModuleList(_tmpOut)
        self.ll_ = nn.ModuleList(_ll_)
        self.tmpOut_ = nn.ModuleList(_tmpOut_)

    def forward(self, x,q=None):
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.r1(x)
        # x = self.maxpool(x)
        x = self.r4(x)
        x = self.r5(x)
        hm_out = []

        for i in range(self.nStack):
            hg = self.hourglass[i](x)
            # hg = self.drop(hg)
            ll = hg
            for j in range(self.nModules):
                ll = self.Residual[i * self.nModules + j](ll)
            ll = self.lin_[i](ll)
            tmpOut = self.tmpOut[i](ll)
            hm_out.append(tmpOut)
            if i < self.nStack - 1:
                ll_ = self.ll_[i](ll)
                tmpOut_ = self.tmpOut_[i](tmpOut)
                x = x + ll_ + tmpOut_
        out = []
        for i in hm_out:
            o = self.softarg(i)
            out.append(o)
        return out