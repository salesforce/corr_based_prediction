
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



ACT = F.relu

class MLPLayer(nn.Module):
    def __init__(self, dim_in=None, dim_out=None, bn=False, act=True, dropout=0., bias=True):
        super(MLPLayer, self).__init__()
        self.act=act
        layer = [nn.Linear(dim_in, dim_out, bias=bias)]
        if bn:
            bn_ = nn.BatchNorm1d(dim_out)
            layer.append(bn_)
            
        self.layer = nn.Sequential(*layer)
        
    def forward(self, x):
        x=self.layer(x)
        if self.act:
            x = ACT((x))
        return x

class CNN(nn.Module):
    def __init__(self, bn=False, affine=True, num_classes=10, bias=False, kernel_size=3, inp_noise=0, VIB=False):
        super(CNN, self).__init__()
        self.VIB = VIB
        nhiddens = [200,400,600,800]
        self.inp_noise = inp_noise

        self.conv1 = nn.Conv2d(3, nhiddens[0], kernel_size, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(nhiddens[0], affine=affine) if bn else nn.Sequential()

        self.conv2 = nn.Conv2d(nhiddens[0], nhiddens[1], 3, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(nhiddens[1], affine=affine)if bn else nn.Sequential()



        self.conv3 = nn.Conv2d(nhiddens[1], nhiddens[2], 3, 1, bias=bias)
        self.bn3 = nn.BatchNorm2d(nhiddens[2], affine=affine) if bn else nn.Sequential()


        self.conv4 = nn.Conv2d(nhiddens[2], nhiddens[3], 3, 1, bias=bias)
        self.bn4 = nn.BatchNorm2d(nhiddens[3], affine=affine) if bn else nn.Sequential()

        nb_filters_cur = nhiddens[3]
        if self.VIB:
            self.mn = MLPLayer(nb_filters_cur, 256, 'none', act=False, bias=bias)
            self.logvar = MLPLayer(nb_filters_cur, 256, 'none', act=False, bias=bias)
            nb_filters_cur = 256


        self.fc = MLPLayer(nb_filters_cur, num_classes, 'none', act=False, bias=bias)

        

    def forward(self, x, ret_hid=False, train=True):
        if x.size()[1]==1: # if MNIST is given, replicate 1 channel to make input have 3 channel
            out = torch.ones(x.size(0), 3, x.size(2), x.size(3)).type('torch.cuda.FloatTensor')
            x = out*x

        if self.inp_noise>0 and train:
            x = x + self.inp_noise*torch.randn_like(x)
        h=self.conv1(x)
        x = F.relu(self.bn1(h))
        x = F.max_pool2d(x, 2, 2)

        x=self.conv2(x)
        x = F.relu(self.bn2(x))

        x=self.conv3(x)
        x = F.relu(self.bn3(x))
        x = F.max_pool2d(x, 2, 2)

        x=self.conv4(x)
        x = F.relu(self.bn4(x))

        x = nn.AvgPool2d(*[x.size()[2]*2])(x)
        x = x.view(x.size()[0], -1)

        if self.VIB:
            mn = self.mn(x)
            logvar = self.logvar(x)
            x = reparameterize(mn,logvar)


        x = self.fc(x)
        if ret_hid:
            return x, h
        elif self.VIB and train:
            return out, mn, logvar
        else:
            return x


class resblock(nn.Module):

    def __init__(self, depth, channels, stride=1, bn='', nresblocks=1.,affine=True, kernel_size=3, bias=True):
        self.depth = depth
        self. channels = channels
        
        super(resblock, self).__init__()
        self.bn1 = nn.BatchNorm2d(depth,affine=affine) if bn else nn.Sequential()
        self.conv2 = (nn.Conv2d(depth, channels, kernel_size=kernel_size, stride=stride, padding=1, bias=bias))
        self.bn2 = nn.BatchNorm2d(channels, affine=affine) if bn else nn.Sequential()

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=1, bias=bias)

        self.shortcut = nn.Sequential()
        if stride > 1 or depth!=channels:
            layers = []
            conv_layer = nn.Conv2d(depth, channels, kernel_size=1, stride=stride, padding=0, bias=bias)
            layers += [conv_layer, nn.BatchNorm2d(channels,affine=affine) if bn else nn.Sequential()]
            self.shortcut = nn.Sequential(*layers)

    def forward(self, x):
        out = ACT(self.bn1(x))
        out = ACT(self.bn2(self.conv2(out)))
        out = (self.conv3(out))
        short = self.shortcut(x)
        out += 1.*short
        return out



class ResNet(nn.Module):
    def __init__(self, depth=56, nb_filters=16, num_classes=10, bn=False, affine=True, kernel_size=3, inp_channels=3, k=1, pad_conv1=0, bias=False, inp_noise=0, VIB=False): # n=9->Resnet-56
        super(ResNet, self).__init__()
        self.inp_noise = inp_noise
        self.VIB = VIB
        nstage = 3 
        
        self.pre_clf=[]

        assert ((depth-2)%6 ==0), 'resnet depth should be 6n+2'
        n = int((depth-2)/6)
        
        nfilters = [nb_filters, nb_filters*k, 2* nb_filters*k, 4* nb_filters*k, num_classes]
        self.nfilters = nfilters
        self.num_classes = num_classes
        self.conv1 = (nn.Conv2d(inp_channels, nfilters[0], kernel_size=kernel_size, stride=1, padding=pad_conv1, bias=bias))
        self.bn1 = nn.BatchNorm2d(nfilters[0], affine=affine) if bn else nn.Sequential()


        nb_filters_prev = nb_filters_cur = nfilters[0]
        for stage in range(nstage):
            nb_filters_cur =  nfilters[stage+1]
            for i in range(n):
                subsample = 1 if (i > 0 or stage == 0) else 2
                layer = resblock(nb_filters_prev, nb_filters_cur, subsample, bn=bn, nresblocks = nstage*n, affine=affine, kernel_size=3, bias=bias)
                self.pre_clf.append(layer)
                nb_filters_prev = nb_filters_cur

        self.pre_clf = nn.Sequential(*self.pre_clf)

        if self.VIB:
            self.mn = MLPLayer(nb_filters_cur, 256, 'none', act=False, bias=bias)
            self.logvar = MLPLayer(nb_filters_cur, 256, 'none', act=False, bias=bias)
            nb_filters_cur = 256

        self.fc = MLPLayer(nb_filters_cur, nfilters[-1], 'none', act=False, bias=bias)
        
    def forward(self, x, ret_hid=False, train=True):
        if x.size()[1]==1: # if MNIST is given, replicate 1 channel to make input have 3 channel
            out = torch.ones(x.size(0), 3, x.size(2), x.size(3)).type('torch.cuda.FloatTensor')
            out = out*x
        else:
            out = x

        if self.inp_noise>0 and train:
            out = out + self.inp_noise*torch.randn_like(out)
        hid = self.conv1(out)

        out = self.bn1(hid)

        out = self.pre_clf(out)

        fc = torch.mean(out.view(out.size(0), out.size(1), -1), dim=2)
        fc = fc.view(fc.size()[0], -1)

        if self.VIB:
            mn = self.mn(fc)
            logvar = self.logvar(fc)
            fc = reparameterize(mn,logvar)

        out = self.fc((fc))


        if ret_hid:
            return out, hid
        elif self.VIB and train:
            return out, mn, logvar
        else:
            return out
    

# Resnet nomenclature: 6n+2 = 3x2xn + 2; 3 stages, each with n number of resblocks containing 2 conv layers each, and finally 2 non-res conv layers
def ResNet_model(bn=False, num_classes=10, depth=56, nb_filters=16, kernel_size=3, inp_channels=3, k=1, pad_conv1=0, affine=True, inp_noise=0, VIB=False):
    return ResNet(depth=depth, nb_filters=nb_filters, num_classes=num_classes, bn=bn, kernel_size=kernel_size, \
                inp_channels=inp_channels, k=k, pad_conv1=pad_conv1, affine=affine, inp_noise=inp_noise, VIB=VIB)


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std
