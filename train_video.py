import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import random
import models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from k4datasets import K4Dataset
import time

parser = argparse.ArgumentParser(description="Deep Multi-Patch Hierarchical Network")
parser.add_argument("-e","--epochs",type = int, default = 4000)
parser.add_argument("-se","--start_epoch",type = int, default = 0)
parser.add_argument("-b","--batchsize",type = int, default = 1)
parser.add_argument("-s","--imagesize",type = int, default = 512)
parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g","--gpu",type=str, default='0')
args = parser.parse_args()

#Hyper Parameters
METHOD = "4KRD"
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
GPU = args.gpu
BATCH_SIZE = args.batchsize
IMAGE_SIZE = args.imagesize

def save_deblur_images(images, iteration, epoch):
    filename = './checkpoints/' + METHOD + "/epoch" + str(epoch) + "/" + "Iter_" + str(iteration) + "_deblur.jpg"
    torchvision.utils.save_image(images, filename)

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0.0, 0.5*math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    print('init GPU')
    if len(GPU.split(',')) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)

    print("init data folders")

    encoder_lv1 = models.Encoder().apply(weight_init).cuda()
    encoder_lv2 = models.Encoder().apply(weight_init).cuda()
    encoder_lv3 = models.Encoder().apply(weight_init).cuda()
    encoder_lv4 = models.Encoder().apply(weight_init).cuda()

    decoder_lv1 = models.Decoder().apply(weight_init).cuda()
    decoder_lv2 = models.Decoder().apply(weight_init).cuda()
    decoder_lv3 = models.Decoder().apply(weight_init).cuda()
    decoder_lv4 = models.Decoder().apply(weight_init).cuda()

    encoder_lv1_optim = torch.optim.Adam(encoder_lv1.parameters(),lr=LEARNING_RATE)
    encoder_lv1_scheduler = StepLR(encoder_lv1_optim,step_size=1000,gamma=0.1)
    encoder_lv2_optim = torch.optim.Adam(encoder_lv2.parameters(),lr=LEARNING_RATE)
    encoder_lv2_scheduler = StepLR(encoder_lv2_optim,step_size=1000,gamma=0.1)
    encoder_lv3_optim = torch.optim.Adam(encoder_lv3.parameters(),lr=LEARNING_RATE)
    encoder_lv3_scheduler = StepLR(encoder_lv3_optim,step_size=1000,gamma=0.1)
    encoder_lv4_optim = torch.optim.Adam(encoder_lv4.parameters(),lr=LEARNING_RATE)
    encoder_lv4_scheduler = StepLR(encoder_lv4_optim,step_size=1000,gamma=0.1)

    decoder_lv1_optim = torch.optim.Adam(decoder_lv1.parameters(),lr=LEARNING_RATE)
    decoder_lv1_scheduler = StepLR(decoder_lv1_optim,step_size=1000,gamma=0.1)
    decoder_lv2_optim = torch.optim.Adam(decoder_lv2.parameters(),lr=LEARNING_RATE)
    decoder_lv2_scheduler = StepLR(decoder_lv2_optim,step_size=1000,gamma=0.1)
    decoder_lv3_optim = torch.optim.Adam(decoder_lv3.parameters(),lr=LEARNING_RATE)
    decoder_lv3_scheduler = StepLR(decoder_lv3_optim,step_size=1000,gamma=0.1)
    decoder_lv4_optim = torch.optim.Adam(decoder_lv4.parameters(),lr=LEARNING_RATE)
    decoder_lv4_scheduler = StepLR(decoder_lv4_optim,step_size=1000,gamma=0.1)

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")):
        encoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")))
        print("load encoder_lv1 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")):
        encoder_lv2.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")))
        print("load encoder_lv2 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")):
        encoder_lv3.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")))
        print("load encoder_lv3 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv4.pkl")):
        encoder_lv4.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv4.pkl")))
        print("load encoder_lv4 success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")):
        decoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")))
        print("load decoder_lv1 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")):
        decoder_lv2.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")))
        print("load decoder_lv2 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv3.pkl")):
        decoder_lv3.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv3.pkl")))
        print("load decoder_lv3 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv4.pkl")):
        decoder_lv4.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv4.pkl")))
        print("load decoder_lv4 success")
    
    if os.path.exists('./checkpoints/' + METHOD) == False:
        os.system('mkdir ./checkpoints/' + METHOD)

    pre_deblur_frame = Variable(torch.full([1, 3, IMAGE_SIZE, IMAGE_SIZE], 0) - 0.5).cuda() # create a tensor,value=0

    for epoch in range(args.start_epoch, EPOCHS):
        encoder_lv1_scheduler.step(epoch)
        encoder_lv2_scheduler.step(epoch)
        encoder_lv3_scheduler.step(epoch)
        encoder_lv4_scheduler.step(epoch)

        decoder_lv1_scheduler.step(epoch)
        decoder_lv2_scheduler.step(epoch)
        decoder_lv3_scheduler.step(epoch)
        decoder_lv4_scheduler.step(epoch)      
        
        print("Training...")
        
        train_dataset = K4Dataset(
            blur_image_files = './datas/4RKD/train_blur_list.txt',
            sharp_image_files = './datas/4RKD/train_sharp_list.txt',
            root_dir = './datas/4RKD',
            crop = True,
            crop_size = IMAGE_SIZE,
            multi_scale = True,
            transform = transforms.Compose([
                transforms.ToTensor()
                ]))

        train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False) # set batch_size=1(BATCH_SIZE), use pre_frame deblur
        start = 0
        
        for iteration, images in enumerate(train_dataloader):
            # torchvision.utils.save_image(pre_deblur_frame.data + 0.5, './checkpoints/K4_video_deblur/epoch_%d_ire_%d.jpg'%(epoch, iteration))
            # pre_deblur_frame = F.interpolate(pre_deblur_frame, scale_factor=0.5, mode='bilinear')

            mse = nn.MSELoss().cuda()
            
            gt_lv1 = Variable(images['sharp_image'] - 0.5).cuda()
            gt_lv2 = Variable(images['sharp_image_s1'] - 0.5).cuda()
            gt_lv3 = Variable(images['sharp_image_s2'] - 0.5).cuda()
            gt_lv4 = Variable(images['sharp_image_s3'] - 0.5).cuda()
            H = gt_lv1.size(2)
            W = gt_lv1.size(3)
            
            images_lv1_ori = Variable(images['blur_image_s1'] - 0.5).cuda()
            if images['video_start']:
                pre_deblur_frame = images_lv1_ori
                print('A new video file start!')
            images_lv1 = torch.cat((pre_deblur_frame, images_lv1_ori), 1)

            images_lv2_ori = Variable(images['blur_image_s2'] - 0.5).cuda()
            pre_frame_s2 = F.interpolate(pre_deblur_frame, scale_factor=0.5, mode='bilinear')
            images_lv2 = torch.cat((pre_frame_s2, images_lv2_ori), 1)

            images_lv3_ori = Variable(images['blur_image_s3'] - 0.5).cuda()
            pre_frame_s3 = F.interpolate(pre_deblur_frame, scale_factor=0.25, mode='bilinear')
            images_lv3 = torch.cat((pre_frame_s3, images_lv3_ori), 1)

            images_lv4_ori = Variable(images['blur_image_s4'] - 0.5).cuda()
            pre_frame_s4 = F.interpolate(pre_deblur_frame, scale_factor=0.125, mode='bilinear')
            images_lv4 = torch.cat((pre_frame_s4, images_lv4_ori), 1)

            images_lv2_1 = images_lv2[:,:,0:int(H/4),:]
            images_lv2_2 = images_lv2[:,:,int(H/4):int(H/2),:]

            images_lv3_1 = images_lv3[:,:,0:int(H/8),0:int(W/8)]
            images_lv3_2 = images_lv3[:,:,0:int(H/8),int(W/8):int(W/4)]
            images_lv3_3 = images_lv3[:,:,int(H/8):int(H/4),0:int(W/8)]
            images_lv3_4 = images_lv3[:,:,int(H/8):int(H/4),int(W/8):int(W/4)]

            images_lv4_1 = images_lv4[:,:,0:int(H/32),0:int(W/16)]
            images_lv4_2 = images_lv4[:,:,int(H/32):int(H/16),0:int(W/16)]
            images_lv4_3 = images_lv4[:,:,0:int(H/32),int(W/16):int(W/8)]
            images_lv4_4 = images_lv4[:,:,int(H/32):int(H/16),int(W/16):int(W/8)]
            images_lv4_5 = images_lv4[:,:,int(H/16):int(H*3/32),0:int(W/16)]
            images_lv4_6 = images_lv4[:,:,int(H*3/32):int(H/8),0:int(W/16)]
            images_lv4_7 = images_lv4[:,:,int(H/16):int(H*3/32),int(W/16):int(W/8)]
            images_lv4_8 = images_lv4[:,:,int(H*3/32):int(H/8),int(W/16):int(W/8)]

            #Lv4
            images_lv4_com = torch.cat((images_lv4_1, images_lv4_2, images_lv4_3, images_lv4_4, images_lv4_5, images_lv4_6, images_lv4_7, images_lv4_8), 0)
            feature_lv4_com, feature_lv4_x1, feature_lv4_x2 = encoder_lv4(images_lv4_com)
            feature_lv4_1 = feature_lv4_com[:BATCH_SIZE*1,:,:,:]
            feature_lv4_2 = feature_lv4_com[BATCH_SIZE*1:BATCH_SIZE*2,:,:,:]
            feature_lv4_3 = feature_lv4_com[BATCH_SIZE*2:BATCH_SIZE*3,:,:,:]
            feature_lv4_4 = feature_lv4_com[BATCH_SIZE*3:BATCH_SIZE*4,:,:,:]
            feature_lv4_5 = feature_lv4_com[BATCH_SIZE*4:BATCH_SIZE*5,:,:,:]
            feature_lv4_6 = feature_lv4_com[BATCH_SIZE*5:BATCH_SIZE*6,:,:,:]
            feature_lv4_7 = feature_lv4_com[BATCH_SIZE*6:BATCH_SIZE*7,:,:,:]
            feature_lv4_8 = feature_lv4_com[BATCH_SIZE*7:,:,:,:]
            feature_lv4_top_left = torch.cat((feature_lv4_1, feature_lv4_2), 2)
            feature_lv4_top_right = torch.cat((feature_lv4_3, feature_lv4_4), 2)
            feature_lv4_bot_left = torch.cat((feature_lv4_5, feature_lv4_6), 2)
            feature_lv4_bot_right = torch.cat((feature_lv4_7, feature_lv4_8), 2)

            x1_lv4_1 = feature_lv4_x1[:BATCH_SIZE*1,:,:,:]
            x1_lv4_2 = feature_lv4_x1[BATCH_SIZE*1:BATCH_SIZE*2,:,:,:]
            x1_lv4_3 = feature_lv4_x1[BATCH_SIZE*2:BATCH_SIZE*3,:,:,:]
            x1_lv4_4 = feature_lv4_x1[BATCH_SIZE*3:BATCH_SIZE*4,:,:,:]
            x1_lv4_5 = feature_lv4_x1[BATCH_SIZE*4:BATCH_SIZE*5,:,:,:]
            x1_lv4_6 = feature_lv4_x1[BATCH_SIZE*5:BATCH_SIZE*6,:,:,:]
            x1_lv4_7 = feature_lv4_x1[BATCH_SIZE*6:BATCH_SIZE*7,:,:,:]
            x1_lv4_8 = feature_lv4_x1[BATCH_SIZE*7:,:,:,:]
            x1_lv4_top_left = torch.cat((x1_lv4_1, x1_lv4_2), 2)
            x1_lv4_top_right = torch.cat((x1_lv4_3, x1_lv4_4), 2)
            x1_lv4_bot_left = torch.cat((x1_lv4_5, x1_lv4_6), 2)
            x1_lv4_bot_right = torch.cat((x1_lv4_7, x1_lv4_8), 2)

            x2_lv4_1 = feature_lv4_x2[:BATCH_SIZE*1,:,:,:]
            x2_lv4_2 = feature_lv4_x2[BATCH_SIZE*1:BATCH_SIZE*2,:,:,:]
            x2_lv4_3 = feature_lv4_x2[BATCH_SIZE*2:BATCH_SIZE*3,:,:,:]
            x2_lv4_4 = feature_lv4_x2[BATCH_SIZE*3:BATCH_SIZE*4,:,:,:]
            x2_lv4_5 = feature_lv4_x2[BATCH_SIZE*4:BATCH_SIZE*5,:,:,:]
            x2_lv4_6 = feature_lv4_x2[BATCH_SIZE*5:BATCH_SIZE*6,:,:,:]
            x2_lv4_7 = feature_lv4_x2[BATCH_SIZE*6:BATCH_SIZE*7,:,:,:]
            x2_lv4_8 = feature_lv4_x2[BATCH_SIZE*7:,:,:,:]
            x2_lv4_top_left = torch.cat((x2_lv4_1, x2_lv4_2), 2)
            x2_lv4_top_right = torch.cat((x2_lv4_3, x2_lv4_4), 2)
            x2_lv4_bot_left = torch.cat((x2_lv4_5, x2_lv4_6), 2)
            x2_lv4_bot_right = torch.cat((x2_lv4_7, x2_lv4_8), 2)

            feature_lv4_top = torch.cat((feature_lv4_top_left, feature_lv4_top_right), 3)
            feature_lv4_bot = torch.cat((feature_lv4_bot_left, feature_lv4_bot_right), 3)
            feature_lv4_top = F.interpolate(feature_lv4_top, scale_factor=2, mode='bilinear')
            feature_lv4_bot = F.interpolate(feature_lv4_bot, scale_factor=2, mode='bilinear')
            # feature_lv4 = torch.cat((feature_lv4_top, feature_lv4_bot), 2)
            feature_lv4_de = torch.cat((feature_lv4_top_left, feature_lv4_top_right, feature_lv4_bot_left, feature_lv4_bot_right), 0)
            x1_lv4_de = torch.cat((x1_lv4_top_left, x1_lv4_top_right, x1_lv4_bot_left, x1_lv4_bot_right), 0)
            x1_lv4_de = F.interpolate(x1_lv4_de, scale_factor=2, mode='bilinear')
            x2_lv4_de = torch.cat((x2_lv4_top_left, x2_lv4_top_right, x2_lv4_bot_left, x2_lv4_bot_right), 0)

            residual_lv4_dee = decoder_lv4(feature_lv4_de, x1_lv4_de, x2_lv4_de)
            residual_lv4_de = torch.cat((residual_lv4_dee, residual_lv4_dee), 1)
            residual_lv4_top_left = residual_lv4_de[:BATCH_SIZE*1,:,:,:]
            residual_lv4_top_right = residual_lv4_de[BATCH_SIZE*1:BATCH_SIZE*2,:,:,:]
            residual_lv4_bot_left = residual_lv4_de[BATCH_SIZE*2:BATCH_SIZE*3,:,:,:]
            residual_lv4_bot_right = residual_lv4_de[BATCH_SIZE*3:,:,:,:]

            image_lv4_decoder_top = torch.cat((residual_lv4_top_left, residual_lv4_top_right), 3)
            image_lv4_decoder_bot = torch.cat((residual_lv4_bot_left, residual_lv4_bot_right), 3)
            image_lv4_decoder = torch.cat((image_lv4_decoder_top, image_lv4_decoder_bot), 2)
            image_lv4_decoder = F.interpolate(image_lv4_decoder, scale_factor=0.5, mode='bilinear')
            loss_lv4 = mse(image_lv4_decoder[:,3:6,:,:], gt_lv4)

            #Lv3
            images_lv3_1_res = residual_lv4_top_left + images_lv3_1
            images_lv3_2_res = residual_lv4_top_right + images_lv3_2
            images_lv3_3_res = residual_lv4_bot_left + images_lv3_3
            images_lv3_4_res = residual_lv4_bot_right + images_lv3_4
            images_lv3_com = torch.cat((images_lv3_1_res, images_lv3_2_res, images_lv3_3_res, images_lv3_4_res), 0)
            feature_lv3_com, feature_lv3_x1, feature_lv3_x2 = encoder_lv3(images_lv3_com)
            feature_lv3_1 = feature_lv3_com[:BATCH_SIZE*1,:,:,:]
            feature_lv3_2 = feature_lv3_com[BATCH_SIZE*1:BATCH_SIZE*2,:,:,:]
            feature_lv3_3 = feature_lv3_com[BATCH_SIZE*2:BATCH_SIZE*3,:,:,:]
            feature_lv3_4 = feature_lv3_com[BATCH_SIZE*3:,:,:,:]
            feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3) + feature_lv4_top
            feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3) + feature_lv4_bot

            x1_lv3_1 = feature_lv3_x1[:BATCH_SIZE*1,:,:,:]
            x1_lv3_2 = feature_lv3_x1[BATCH_SIZE*1:BATCH_SIZE*2,:,:,:]
            x1_lv3_3 = feature_lv3_x1[BATCH_SIZE*2:BATCH_SIZE*3,:,:,:]
            x1_lv3_4 = feature_lv3_x1[BATCH_SIZE*3:,:,:,:]
            x1_lv3_top = torch.cat((x1_lv3_1, x1_lv3_2), 3)
            x1_lv3_bot = torch.cat((x1_lv3_3, x1_lv3_4), 3)

            x2_lv3_1 = feature_lv3_x2[:BATCH_SIZE*1,:,:,:]
            x2_lv3_2 = feature_lv3_x2[BATCH_SIZE*1:BATCH_SIZE*2,:,:,:]
            x2_lv3_3 = feature_lv3_x2[BATCH_SIZE*2:BATCH_SIZE*3,:,:,:]
            x2_lv3_4 = feature_lv3_x2[BATCH_SIZE*3:,:,:,:]
            x2_lv3_top = torch.cat((x2_lv3_1, x2_lv3_2), 3)
            x2_lv3_bot = torch.cat((x2_lv3_3, x2_lv3_4), 3)

            feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
            feature_lv3 = F.interpolate(feature_lv3, scale_factor=2, mode='bilinear')
            feature_lv3_de = torch.cat((feature_lv3_top, feature_lv3_bot), 0)
            x1_lv3_de = torch.cat((x1_lv3_top, x1_lv3_bot), 0)
            x1_lv3_de = F.interpolate(x1_lv3_de, scale_factor=2, mode='bilinear')
            x2_lv3_de = torch.cat((x2_lv3_top, x2_lv3_bot), 0)

            residual_lv3_dee = decoder_lv3(feature_lv3_de, x1_lv3_de, x2_lv3_de)
            residual_lv3_de = torch.cat((residual_lv3_dee, residual_lv3_dee), 1)
            residual_lv3_top = residual_lv3_de[:BATCH_SIZE*1,:,:,:]
            residual_lv3_bot = residual_lv3_de[BATCH_SIZE*1:,:,:,:]

            image_lv3_decoder = torch.cat((residual_lv3_top, residual_lv3_bot), 2)
            image_lv3_decoder = F.interpolate(image_lv3_decoder, scale_factor=0.5, mode='bilinear')
            loss_lv3 = mse(image_lv3_decoder[:,3:6,:,:], gt_lv3)

            #Lv2
            images_lv2_1_res = residual_lv3_top + images_lv2_1
            images_lv2_2_res = residual_lv3_bot + images_lv2_2
            images_lv2_com = torch.cat((images_lv2_1_res, images_lv2_2_res), 0)
            feature_lv2_com,  feature_lv2_x1, feature_lv2_x2= encoder_lv2(images_lv2_com)
            feature_lv2_1 = feature_lv2_com[:BATCH_SIZE*1,:,:,:]
            feature_lv2_2 = feature_lv2_com[BATCH_SIZE*1:,:,:,:]
            feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3

            x1_lv2_1 = feature_lv2_x1[:BATCH_SIZE*1,:,:,:]
            x1_lv2_2 = feature_lv2_x1[BATCH_SIZE*1:,:,:,:]
            x1_lv2 = torch.cat((x1_lv2_1, x1_lv2_2), 2)
            x1_lv2 = F.interpolate(x1_lv2, scale_factor=2, mode='bilinear')

            x2_lv2_1 = feature_lv2_x2[:BATCH_SIZE*1,:,:,:]
            x2_lv2_2 = feature_lv2_x2[BATCH_SIZE*1:,:,:,:]
            x2_lv2 = torch.cat((x2_lv2_1, x2_lv2_2), 2)

            residual_lv2_dee = decoder_lv2(feature_lv2, x1_lv2, x2_lv2)
            residual_lv2 = torch.cat((residual_lv2_dee, residual_lv2_dee), 1)
            feature_lv2 = F.interpolate(feature_lv2, scale_factor=2, mode='bilinear')

            residual_lv2_loss = F.interpolate(residual_lv2, scale_factor=0.5, mode='bilinear')
            loss_lv2 = mse(residual_lv2_loss[:,3:6,:,:], gt_lv2)

            #Lv1
            feature_lv1, feature_lv1_x1, feature_lv1_x2 = encoder_lv1(images_lv1 + residual_lv2)
            feature_lv1_x1 = F.interpolate(feature_lv1_x1, scale_factor=2, mode='bilinear')
            deblur_image = decoder_lv1(feature_lv1 + feature_lv2, feature_lv1_x1, feature_lv1_x2)

            image_deblur = F.interpolate(deblur_image, scale_factor=0.5, mode='bilinear')
            pre_deblur_frame.data = image_deblur.data

            loss_lv1 = mse(image_deblur, gt_lv1)

            # calculate tv(total variation) loss
            diff_i = torch.sum(torch.abs(image_deblur[:, :, :, 1:] - image_deblur[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(image_deblur[:, :, 1:, :] - image_deblur[:, :, :-1, :]))
            tv_loss = diff_i + diff_j

            loss = 0.7*loss_lv1 + 0.15*loss_lv2 + 0.1*loss_lv3 + 0.05*loss_lv4 + 0.000000035*tv_loss
            # loss = 0.7*loss_lv1 + 0.15*loss_lv2 + 0.1*loss_lv3 + 0.05*loss_lv4
            
            encoder_lv1_optim.zero_grad()
            encoder_lv2_optim.zero_grad()
            encoder_lv3_optim.zero_grad()
            encoder_lv4_optim.zero_grad()

            decoder_lv1_optim.zero_grad()
            decoder_lv2_optim.zero_grad()
            decoder_lv3_optim.zero_grad()
            decoder_lv4_optim.zero_grad()
            
            loss.backward()

            encoder_lv1_optim.step()
            encoder_lv2_optim.step()
            encoder_lv3_optim.step()
            encoder_lv4_optim.step()

            decoder_lv1_optim.step()
            decoder_lv2_optim.step()
            decoder_lv3_optim.step()
            decoder_lv4_optim.step()
            
            if (iteration+1)%10 == 0:
                stop = time.time()
                print("4KRD_deblur epoch:", epoch, "iteration:", iteration+1, "loss:%.4f"%loss.item(), 'time:%.4f'%(stop-start))
                start = time.time()
                
            if iteration != 0 and (iteration)%10000==0:
                model_path = './checkpoints/' + METHOD + '/epoch_%04d'%(epoch) + '_Ite_%04d'%(iteration)
                if not os.path.exists(model_path):
                	os.system('mkdir %s'%(model_path))

                torch.save(encoder_lv1.state_dict(),'%s/encoder_lv1.pkl'%(model_path))
                torch.save(encoder_lv2.state_dict(),'%s/encoder_lv2.pkl'%(model_path))
                torch.save(encoder_lv3.state_dict(),'%s/encoder_lv3.pkl'%(model_path))
                torch.save(encoder_lv4.state_dict(),'%s/encoder_lv4.pkl'%(model_path))
                torch.save(decoder_lv1.state_dict(),'%s/decoder_lv1.pkl'%(model_path))
                torch.save(decoder_lv2.state_dict(),'%s/decoder_lv2.pkl'%(model_path))
                torch.save(decoder_lv3.state_dict(),'%s/decoder_lv3.pkl'%(model_path))
                torch.save(decoder_lv4.state_dict(),'%s/decoder_lv4.pkl'%(model_path))

        model_path = './checkpoints/' + METHOD + '/epoch_%04d'%(epoch)
        if not os.path.exists(model_path):
            os.system('mkdir %s'%(model_path))
        torch.save(encoder_lv1.state_dict(),'%s/encoder_lv1.pkl'%(model_path))
        torch.save(encoder_lv2.state_dict(),'%s/encoder_lv2.pkl'%(model_path))
        torch.save(encoder_lv3.state_dict(),'%s/encoder_lv3.pkl'%(model_path))
        torch.save(encoder_lv4.state_dict(),'%s/encoder_lv4.pkl'%(model_path))
        torch.save(decoder_lv1.state_dict(),'%s/decoder_lv1.pkl'%(model_path))
        torch.save(decoder_lv2.state_dict(),'%s/decoder_lv2.pkl'%(model_path))
        torch.save(decoder_lv3.state_dict(),'%s/decoder_lv3.pkl'%(model_path))
        torch.save(decoder_lv4.state_dict(),'%s/decoder_lv4.pkl'%(model_path))
           

if __name__ == '__main__':
    main()

        

        

