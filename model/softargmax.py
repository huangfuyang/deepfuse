from torch.nn import Module
from torch.nn import functional as F
from params import ARGMAXK
import torch
from torch.autograd import Variable


class Softargmax3D(Module):
    def __init__(self,size,batch,channel):
        super(Softargmax3D, self).__init__()
        r = self.size = size
        b = self.b = batch
        c = self.c = channel
        i = torch.arange(0, r, 1.0).unsqueeze(-1).cuda()
        self.xi = i.repeat(1, r * r).view(-1).repeat(b * c).view(b, c, -1)
        self.yi = i.repeat(r, r).view(-1).repeat(b * c).view(b, c, -1)
        self.zi = i.view(1, r).repeat(r * r, 1).view(-1).repeat(b * c).view(b, c, -1)

    def forward(self, input):
        s = input.size()
        input = input*ARGMAXK
        b = s[0]
        c = s[1]
        r = s[2]

        assert self.size == r
        if b < self.b:
            xi = self.xi[:b,:,:]
            yi = self.yi[:b,:,:]
            zi = self.zi[:b,:,:]

        else:
            xi = self.xi
            yi = self.yi
            zi = self.zi
        soft = F.softmax(input.view(b, c, -1), dim=2)
        x = (soft * xi).sum(dim=2)
        y = (soft * yi).sum(dim=2)
        z = (soft * zi).sum(dim=2)
        return torch.cat((x,y,z)).view(3,-1).transpose(0, 1).contiguous().view(b,-1)

    # def extra_repr(self):
    #     if self.scale_factor is not None:
    #         info = 'scale_factor=' + str(self.scale_factor)
    #     else:
    #         info = 'size=' + str(self.size)
    #     info += ', mode=' + self.mode
    #     return info


if __name__ == '__main__':
    t = Variable(torch.randn(1,1,3,3,3).cuda().abs())
    # print t
    import helper
    t = torch.from_numpy(helper.Gaussian3D(5)).unsqueeze(0).unsqueeze(0).cuda()

    print t
    so = Softargmax3D(5,1,1)
    print so(t)
    ARGMAXK = 1
    print so(t)
