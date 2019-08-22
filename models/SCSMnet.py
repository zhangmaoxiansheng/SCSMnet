from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *
from .submodule_DSR import *
import matplotlib.pyplot as plt
#import sparseconvnet as scn
#################
#feature5 = 2048 H/64 too deep and useless
#feature4 = 1024 H/32
#feature3 = 512 H/16
#feature2 = 256 H/8
#feature1 = 64 H/4
#feature0 = 64 H/2
#################

class SCSMnet(nn.Module):
    
    def __init__(self, max_disp,train_stage,distiall_max):
        super(SCSMnet, self).__init__()
        self.max_disp = max_disp
        self.train_stage = train_stage
        layer = [1,2,2,1]
        self.feature_extraction = feature_extraction(layer)
        self.decoder_16 = decoderBlock_dense(5,32,16)
        self.decoder_8 = decoderBlock_dense(5,32+8,16)
        self.decoder_4 = decoderBlock_dense(5,32+8,16,up=False)
        
        self.disp_reg_4 = disparityregression(self.max_disp//4,4)
        self.cost_refine_4 = cost_volume_refine()
        self.disp_reg_8 = disparityregression(self.max_disp//8,8)
        #self.cost_refine_8 = cost_volume_refine(self.max_disp//8)
        self.disp_reg_16 = disparityregression(self.max_disp//16,16)
        #self.cost_refine_16 = cost_volume_refine(self.max_disp//16)
        if self.train_stage == "distill":
            self.max_range = distiall_max
            layer = [3,4,6,3]
            self.DSR_feature_extraction = DSR_feature_extraction(layer)
            self.DSR_disp_pre_extraction = disp_pre_extraction()
            self.DSR_image_pre_extraction = image_pre_extraction()
        
            self.width = 512
        
            self.DSR_up6 = DSR_upconv(2048, 512)#2048->512 H/32
        
            self.DSR_up5 = DSR_upconv(1536, 256)#1024+512=1536 H/16
        
            self.DSR_up4 = DSR_upconv(768, 128)#512+256=768 H/8
            self.DSR_disp4 = DSR_get_disp(128)
        
            self.DSR_up3 = DSR_upconv(384, 64)#256+128=384 H/4
            self.DSR_disp3 = DSR_get_disp(64+1)
        
            self.DSR_up2 = DSR_upconv(128,32)#64+64=128 H/2
            self.DSR_disp2 = DSR_get_disp(32+1)
        
            self.DSR_up1 = DSR_upconv(96,32)#64+32=96 H/2
            self.DSR_disp1 = DSR_get_disp(32+1)
    def cost_volume(self, refimg_fea, targetimg_fea, maxdisp, leftview=True):
        '''
        diff cost volume
        '''
        width = refimg_fea.shape[-1]
        #mask_channel = refimg_fea.shape[1]
        # print(refimg_fea.size()[0])
        # print(mask_channel)
        # print(refimg_fea.size())
        cost = Variable(torch.cuda.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1], maxdisp,  refimg_fea.size()[2],  refimg_fea.size()[3]).fill_(0.))
        
        #full_mask = Variable(torch.cuda.FloatTensor(refimg_fea.size()[0],refimg_fea.size()[1],refimg_fea.size()[2],refimg_fea.size()[3]).fill_(20),requires_grad=False)

        for i in range(maxdisp):
            feata = refimg_fea[:,:,:,i:width]
            featb = targetimg_fea[:,:,:,:width-i]
            diff = torch.abs(feata - featb)
            #mask = full_mask[:,:,:,i:width] - diff
            #mask = torch.clamp(mask,0,20)
            #fullcost = feata * featb
            # concat
            if leftview:
                cost[:, :refimg_fea.size()[1], i, :,i:] = diff
            else:
                cost[:, :refimg_fea.size()[1], i, :,:width-i] = diff
        cost = cost.contiguous()

        return cost

    def forward(self,left,right):
        # print(image.size())
        left_features = self.feature_extraction(left)#origin size 64D test version
        right_features = self.feature_extraction(right)
        

        cost_volume_16 = self.cost_volume(left_features[3],right_features[3],self.max_disp//16)
        cost_volume_8 = self.cost_volume(left_features[2],right_features[2],self.max_disp//8)
        cost_volume_4 = self.cost_volume(left_features[1],right_features[1],self.max_disp//4)
        #print(cost_volume_16.size())
        cost_volume_16x2,cost_16 = self.decoder_16(cost_volume_16)
        cost_16 = F.softmax(-cost_16,1)
        disp_16,vol_16 = self.disp_reg_16(cost_16)
        cost_volume_8 = torch.cat((cost_volume_16x2,cost_volume_8),1)
        

        cost_volume_8x2,cost_8 = self.decoder_8(cost_volume_8)
        cost_8 = F.softmax(-cost_8,1)
        disp_8,vol_8 = self.disp_reg_8(cost_8)
        cost_volume_4 = torch.cat((cost_volume_4,cost_volume_8x2),1)
        disp_8U = F.interpolate(disp_8,[left_features[1].size()[2],left_features[1].size()[3]],mode='bilinear').squeeze()
        cost_volume_4x2,cost_4 = self.decoder_4(cost_volume_4)
        cost_4 = self.cost_refine_4(-cost_4)
        disp_4,vol_4 = self.disp_reg_4(cost_4)
        disp_16 = F.interpolate(disp_16,[left.size()[2],left.size()[3]],mode='bilinear')
        disp_8 = F.interpolate(disp_8,[left.size()[2],left.size()[3]],mode='bilinear')
        disp_4 = F.interpolate(disp_4,[left.size()[2],left.size()[3]],mode='bilinear')
        if self.train_stage == 'distill':
            lr = disp_4
            lr_pyramid = scale_pyramid(lr)
            lr_pyramid = [torch.squeeze(d,1) for d in lr_pyramid]
            image = left
            image_pre_feature = self.DSR_image_pre_extraction(image)
            disp_pre_feature = self.DSR_disp_pre_extraction(lr)
            image_feature     = self.DSR_feature_extraction(image_pre_feature)
            disp_feature  = self.DSR_feature_extraction(disp_pre_feature)
            iconv6 = self.DSR_up6(image_feature[5] + disp_feature[5])#H/32
            iconv5 = self.DSR_up5(torch.cat((iconv6, image_feature[4] + disp_feature[4]),1))#H/16
            iconv4 = self.DSR_up4(torch.cat((image_feature[3] + disp_feature[3], iconv5), 1))#H/8
            output_disp4 = self.DSR_disp4(iconv4, self.max_range)# + lr_pyramid[3]
            up_disp4 = F.interpolate(output_disp4,scale_factor=2,mode='bilinear',align_corners=True)
            iconv3 = self.DSR_up3(torch.cat((image_feature[2] + disp_feature[2], iconv4), 1))#H/4
            output_disp3 = self.DSR_disp3(torch.cat((iconv3,up_disp4),1), self.max_range)# + lr_pyramid[2]
            up_disp3 = F.interpolate(output_disp3,scale_factor=2,mode='bilinear',align_corners=True)
            iconv2 = self.DSR_up2(torch.cat((image_feature[1] + disp_feature[1], iconv3), 1))#H/2
            output_disp2 = self.DSR_disp2(torch.cat((iconv2,up_disp3),1), self.max_range)# + lr_pyramid[1]
            up_disp2 = F.interpolate(output_disp2,scale_factor=2,mode='bilinear',align_corners=True)
            iconv1 = self.DSR_up1(torch.cat((image_feature[0] + disp_feature[0],iconv2),1))#H        
            output_disp1 = self.DSR_disp1(torch.cat((iconv1,up_disp2),1), self.max_range)# + lr_pyramid[0]
            return output_disp1, output_disp2, output_disp3, output_disp4,lr_pyramid

        return disp_4,disp_8,disp_16,cost_4,cost_8,cost_16
