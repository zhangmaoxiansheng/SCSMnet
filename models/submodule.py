from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
#import sparseconvnet as scn
class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                        nn.BatchNorm2d(int(n_filters)),
                                        nn.LeakyReLU(0.1, inplace=True),)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                        nn.LeakyReLU(0.1, inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNorm, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)


        if with_bn:
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.BatchNorm2d(int(n_filters)),)
        else:
            self.cb_unit = nn.Sequential(conv_mod,)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs

def conv3x3(in_channel, out_channel, stride = 1):
    return nn.Conv2d(in_channel, out_channel, kernal_size = 3, stride = stride, padding = 1, bias = False)

def convbn_3d(in_channel, out_channel, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                        nn.BatchNorm3d(out_channel))
class bottleneck_layer(nn.Module):
    expansion = 4#class attribute
    def __init__(self, inplanes, planes , stride = 1, downsample = None):
        super(bottleneck_layer, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None,dilation=1):
        super(residualBlock, self).__init__()

        if dilation > 1:
            padding = dilation
        else:
            padding = 1
        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, padding, bias=False,dilation=dilation)
        self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #out = self.relu(out)
        return out

class feature_extraction(nn.Module):
    def __init__(self, layers):
        super(feature_extraction,self).__init__()
        self.inplanes = 64
        self.layers = layers
        block = residualBlock
        
        self.conv1 = conv2DBatchNormRelu(3,32,7,2,3)#H/2,W/2
        self.conv2 = conv2DBatchNormRelu(32,64,3,1,1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#H/4,W/4
        self.layer1 = self._make_layer(block, 64, self.layers[0], stride=1)#H/4,W/4
        self.layer2 = self._make_layer(block, 64, self.layers[1], stride=2)#H/8,W/8
        self.layer3 = self._make_layer(block, 128, self.layers[2], stride=2)#H/16,W/16
        self.layer4 = self._make_layer(block, 128, self.layers[3], stride=2)#H/32,W/32
        self.pyramidPooling = pyramidPooling(128)

        self.upconv4 = upconv(128,64)#H/16,W/16
        self.iconv4 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=64,
                                                padding=1, stride=1, bias=False)
        self.upconv3 = upconv(64,64)#H/8,W/8
        self.iconv3 = conv2DBatchNormRelu(in_channels=64+64, k_size=3, n_filters=64,
                                                padding=1, stride=1, bias=False)
        self.upconv2 = upconv(64,64)#H/4,W/4
        self.iconv2 = conv2DBatchNormRelu(in_channels=64+64, k_size=3, n_filters=64,
                                                padding=1, stride=1, bias=False)
        self.upconv1 = upconv(64,32)#H/2,W/2
        self.iconv1 = conv2DBatchNormRelu(in_channels=64+32, k_size=3, n_filters=32,
                                                padding=1, stride=1, bias=False)
        
        self.output4 = nn.Sequential(conv2DBatchNormRelu(in_channels=64,k_size=1,n_filters=32,padding=0,stride=1,bias=False),
                                    pyramidPooling(32))#H/16 W/16 32
        
        self.output3 = nn.Sequential(conv2DBatchNormRelu(in_channels=64,k_size=1,n_filters=32,padding=0,stride=1,bias=False),
                                    pyramidPooling(32))#H/8 W/8 32
        self.output2 = nn.Sequential(conv2DBatchNormRelu(in_channels=64,k_size=1,n_filters=32,padding=0,stride=1,bias=False),
                                    pyramidPooling(32))#H/4 W/4 32
        self.output1 = pyramidPooling(32)#H/2 W/2 32
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: 
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion#every time the output channel is input channel * 4
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        origin_size = x.size()
        x1 = self.conv1(x)
        x1 = self.conv2(x1)#H/2 W/2 64
        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)#H/4 W/4 64
        x3 = self.layer2(x2)#H/8 W/8 64
        x4 = self.layer3(x3)#H/16 W/16 128
        x5 = self.layer4(x4)#H/32 W/32 128

        x5 = self.pyramidPooling(x5)
        up4 = self.upconv4(x5)#H/16 W/16 64
        concat4 = torch.cat((up4,x4),1)#h/16 w/16 192
        iconv4 = self.iconv4(concat4)#H/16 w/16 64

        up3 = self.upconv3(iconv4)#H/8 w/8 64
        concat3 = torch.cat((up3,x3),1)#h/8 w/8 128
        iconv3 = self.iconv3(concat3)#h/8 w/8 64

        up2 = self.upconv2(iconv3)#h/4 w/4 64
        concat2 = torch.cat((up2,x2),1)#h/4 w/4 128
        iconv2 = self.iconv2(concat2)#h/4 w/4 64

        up1 = self.upconv1(iconv2)#h/2 w/2 32
        concat1 = torch.cat((up1,x1),1)#h/2 w/2 96
        iconv1 = self.iconv1(concat1)#h/2 w/2 32

        output4 = self.output4(iconv4)#H/16 W/16 32
        output3 = self.output3(iconv3)#H/8 W/8 32
        output2 = self.output2(iconv2)#H/4 W/4 32
        output1 = self.output1(iconv1)#H/2 W/2 32
        
        return output1,output2,output3,output4

class pyramidPooling(nn.Module):

    def __init__(self, in_channels, with_bn=True):
        super(pyramidPooling, self).__init__()
        bias = not with_bn
        self.convs = []
        for i in range(4):
            self.convs.append(conv2DBatchNormRelu(in_channels, in_channels, 1, 1, 0, bias=bias, with_bn=with_bn))
        self.path_module_list = nn.ModuleList(self.convs)
    def forward(self, x):
        h, w = x.shape[2:]

        k_sizes = []
        strides = []
        for pool_size in np.linspace(1,min(h,w)//2,4,dtype=int):
            k_sizes.append((int(h/pool_size), int(w/pool_size)))
            strides.append((int(h/pool_size), int(w/pool_size)))
        k_sizes = k_sizes[::-1]
        strides = strides[::-1]
        pp_sum = x

        for i, module in enumerate(self.path_module_list):
            out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
            out = module(out)
            out = F.interpolate(out, size=(h,w), mode='bilinear')
            pp_sum = pp_sum + 0.25*out
        pp_sum = F.relu(pp_sum/2.,inplace=True)

        return pp_sum


def Conv3d(in_planes, out_planes, kernel_size, stride, pad,bias=False):
    if bias:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias))
    else:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias),
                        nn.BatchNorm3d(out_planes))
class Conv3dBlock_dense(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size=1,stride=1,pad=0,bias = False):
        super(Conv3dBlock_dense, self).__init__()
        self.conv1 = nn.Conv3d(in_plane,out_plane,kernel_size,stride,padding=pad,bias=bias)
        self.conv2 = nn.Conv3d(out_plane,out_plane,kernel_size,stride,padding=pad,bias=bias)
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        out = F.relu(x1 + x2,inplace=True)
        return out
def Conv3d_sparse(in_plane,out_plane,kernel=3,bias=False):
    return scn.Sequential(scn.SubmanifoldConvolution(3,in_plane,out_plane,kernel,bias))
def Conv3dBlock_sparse(in_plane,out_plane):
    m = scn.Sequential()
    m.add(scn.ConcatTable().add(
        scn.Sequential().add(Conv3d_sparse(in_plane,out_plane)).add(Conv3d_sparse(in_plane,out_plane))).add(
            scn.NetworkInNetwork(in_plane,out_plane,False)))
    m.add(scn.AddTable())
    return m

class decoderBlock_sparse(nn.Module):
    def __init__(self, nconvs, inchannelF,channelF,stride=(1,1,1),up=False, nstride=1,pool=False):
        nn.Module.__init__(self)
        self.pool=pool
        #stride = [stride]*nstride + [(1,1,1)] * (nconvs-nstride)#2*[(1,1,1)]=[(1,1,1),(1,1,1)]
        self.convs = [Conv3dBlock_sparse(inchannelF,channelF)]
        for i in range(1,nconvs):
            self.convs.append(Conv3dBlock_dense(channelF,channelF))#actually the stride is the same....
        self.convs = scn.Sequential(*self.convs)
        self.classify = nn.Sequential(Conv3d_sparse(channelF, channelF),
                                       scn.ReLU(),
                                       Conv3d_sparse(channelF, 1, bias=True))

        self.up = False
        if up:
            self.up = True
            self.up = scn.sequential(scn.UnPooling(3,2,2),Conv3d_sparse(in_plane, in_plane//2),scn.ReLU())
        
        if pool:
            self.pool_convs = [Conv3d_sparse(channelF,channelF,1),Conv3d_sparse(channelF,channelF,1),Conv3d_sparse(channelF,channelF,1),Conv3d_sparse(channelF,channelF,1)]

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #torch.nn.init.xavier_uniform(m.weight)
                #torch.nn.init.constant(m.weight,0.001)
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
    def forward(self,fvl):
        # left
        fvl = self.convs(fvl)
        # pooling PSP module
        if self.pool:
            fvl_out = fvl
            _,_,d,h,w=fvl.shape
            for i,pool_size in enumerate(np.linspace(1,min(d,h,w)//2,4,dtype=int)):
                kernel_size = (int(d/pool_size), int(h/pool_size), int(w/pool_size))
                out = scn.AveragePooling(3,kernel_size,kernel_size)(fvl)
                out = self.pool_convs[i](out)
                out = F.interpolate(out, size=(d,h,w), mode='trilinear')#????
                fvl_out = fvl_out + 0.25*out
            fvl = F.relu(fvl_out/2.,inplace=True)

        if self.training:
            # classification and output the cost volume for next stage
            costl = self.classify(fvl)
            if self.up:
                fvl = self.up(fvl)
        else:
            # classification
            if self.up:
                fvl = self.up(fvl)
                costl=fvl
            else:
                costl = self.classify(fvl)

        return fvl,costl.squeeze(1)


class decoderBlock_dense(nn.Module):
    def __init__(self, nconvs,inchannelF,channelF,stride=(1,1,1),up=True, nstride=1,pool=True):
        super(decoderBlock_dense, self).__init__()
        self.pool=pool
        #stride = [stride]*nstride + [(1,1,1)] * (nconvs-nstride)#2*[(1,1,1)]=[(1,1,1),(1,1,1)]
        self.convs = [Conv3dBlock_dense(inchannelF,channelF,stride=stride)]
        for i in range(1,nconvs):
            self.convs.append(Conv3dBlock_dense(channelF,channelF, stride=stride))#actually the stride is the same....
        self.convs = nn.Sequential(*self.convs)

        self.classify = nn.Sequential(Conv3d(channelF, channelF, 3, (1,1,1), 1),
                                       nn.ReLU(inplace=True),
                                       Conv3d(channelF, 1, 3, (1,1,1),1,bias=True))

        self.up = False
        if up:
            self.up = True
            self.up = nn.Sequential(nn.Upsample(scale_factor=2,mode='trilinear'),
                                 Conv3d(channelF, channelF//2, 3, (1,1,1),1,bias=False),
                                 nn.ReLU(inplace=True))

        if pool:
            self.pool_convs = torch.nn.ModuleList([Conv3d(channelF, channelF, 1, (1,1,1), 0),
                               Conv3d(channelF, channelF, 1, (1,1,1), 0),
                               Conv3d(channelF, channelF, 1, (1,1,1), 0),
                               Conv3d(channelF, channelF, 1, (1,1,1), 0)])

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #torch.nn.init.xavier_uniform(m.weight)
                #torch.nn.init.constant(m.weight,0.001)
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
    def forward(self,fvl):
        # left
        
        fvl = self.convs(fvl)
        # pooling PSP module
        if self.pool:
            fvl_out = fvl
            _,_,d,h,w=fvl.shape
            for i,pool_size in enumerate(np.linspace(1,min(d,h,w)//2,4,dtype=int)):
                kernel_size = (int(d/pool_size), int(h/pool_size), int(w/pool_size))
                out = F.avg_pool3d(fvl, kernel_size, stride=kernel_size)       
                out = self.pool_convs[i](out)
                out = F.interpolate(out, size=(d,h,w), mode='trilinear')
                fvl_out = fvl_out + 0.25*out
            fvl = F.relu(fvl_out/2.,inplace=True)

        if self.training:
            # classification and output the cost volume for next stage
            costl = self.classify(fvl)
            if self.up:
                fvl = self.up(fvl)
        else:
            # classification
            if self.up:
                fvl = self.up(fvl)
                costl=fvl
            else:
                costl = self.classify(fvl)

        return fvl,costl.squeeze(1)


class upconv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(upconv,self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = F.interpolate(x, scale_factor=2,mode='bilinear')
        x = self.up(x)
        return x


def scale_pyramid(img, num_scale = 5):   
    
    scaled_imgs = [img.squeeze(1)]
    #print(shape)
    for i in range(num_scale - 1):
        scaled_imgs.append(F.interpolate(img.unsqueeze(1),scale_factor=0.5**(i+1),mode='bilinear').squeeze(1))

    return scaled_imgs

class disparityregression(nn.Module):
    def __init__(self, maxdisp,div):
        super(disparityregression, self).__init__()
        #self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)
        self.register_buffer('disp',torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])))
        self.div = div
    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        #print(disp.size())
        #print(x.size())
        out = torch.sum(x*disp,1) * self.div
        out = out.unsqueeze(1)
        return out

def Gpu_Memory(device=None):
    return torch.cuda.memory_allocated(device=device)/1000/1000
def SSIM(x,y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)
def warp(x, disp):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        #print(disp.size())
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        #grid = torch.cat((xx,yy),1).float()
        x_grid = xx.float()
        y_grid = yy.float()
        if x.is_cuda:
            x_grid = x_grid.cuda()
            y_grid = y_grid.cuda()
        vx_grid = Variable(x_grid) + disp
        vy_grid = Variable(y_grid)
        # scale grid to [-1,1]
        vx_grid = 2.0 * vx_grid / max(W-1,1)-1.0
        vy_grid = 2.0 * vy_grid / max(H-1,1)-1.0
        vgrid = torch.cat((vx_grid,vy_grid),1)
        # vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:] / max(W-1,1)-1.0
        # vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:] / max(H-1,1)-1.0
        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        
        return output
