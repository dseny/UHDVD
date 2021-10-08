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
import time
from PIL import Image

parser = argparse.ArgumentParser(description="Ultra High-Definition Video Deblurring Network")
parser.add_argument("-e","--epochs",type = int, default = 2600)
parser.add_argument("-se","--start_epoch",type = int, default = 0)
parser.add_argument("-b","--batchsize",type = int, default = 1)
parser.add_argument("-s","--imagesize",type = int, default = 256)
parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g","--gpu",type=str, default='0')
args = parser.parse_args()

#Hyper Parameters
DATASET = '4KRD_TEST'
METHOD = "4KRD"
SAMPLE_DIR = "./test_set/%s"%(DATASET)
EXPDIR = "deblur_4KRD"
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
GPU = args.gpu
BATCH_SIZE = args.batchsize
IMAGE_SIZE = args.imagesize

def save_images(images, video, name):
    file_path = os.path.join('./test_set/%s/'%(DATASET), video, EXPDIR)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    torchvision.utils.save_image(images, os.path.join(file_path, name))

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5*math.sqrt(2. / n))
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

    print("init models %s"%METHOD)

    encoder_lv1 = models.Encoder().apply(weight_init).cuda()
    encoder_lv2 = models.Encoder().apply(weight_init).cuda()
    encoder_lv3 = models.Encoder().apply(weight_init).cuda()
    encoder_lv4 = models.Encoder().apply(weight_init).cuda()

    decoder_lv1 = models.Decoder().apply(weight_init).cuda()
    decoder_lv2 = models.Decoder().apply(weight_init).cuda()
    decoder_lv3 = models.Decoder().apply(weight_init).cuda()
    decoder_lv4 = models.Decoder().apply(weight_init).cuda()

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
     
    iteration = 0.0
    test_time = 0.0

    video_names = os.listdir(SAMPLE_DIR)

    for video_name in video_names:
        if os.path.isfile(os.path.join(SAMPLE_DIR, video_name)):
            continue

        video_path = os.path.join(SAMPLE_DIR, video_name, 'blurry')

        frames_list = sorted(os.listdir(video_path))
        pre_deblur_frame = Variable(torch.full([1, 3, 2160, 3840], 0) - 0.5).cuda()

        for i in range(len(frames_list)):
            with torch.no_grad():
                images_ori = transforms.ToTensor()(Image.open(video_path + '/' + frames_list[i]).convert('RGB'))
                H_ori = images_ori.size(1)
                W_ori = images_ori.size(2)
                H = int(math.ceil(H_ori/128)*128)
                W = int(math.ceil(W_ori/128)*128)
                # print('H: %d, W: %d'%(H, W))
                pre_deblur_frame = F.interpolate(pre_deblur_frame, (H, W), mode='bilinear')

                images_ori = transforms.ToPILImage()(images_ori)
                blur_image_s1 = transforms.Resize([H, W])(images_ori)
                blur_image_s2 = transforms.ToTensor()(transforms.Resize([int(H/2), int(W/2)])(blur_image_s1))
                blur_image_s3 = transforms.ToTensor()(transforms.Resize([int(H/4), int(W/4)])(blur_image_s1))
                blur_image_s4 = transforms.ToTensor()(transforms.Resize([int(H/8), int(W/8)])(blur_image_s1))
                blur_image_s1 = transforms.ToTensor()(blur_image_s1)

                images_lv1_ori = Variable(blur_image_s1 - 0.5).unsqueeze(0).cuda()
                if i == 0:
                    pre_deblur_frame = images_lv1_ori
                images_lv1 = torch.cat((pre_deblur_frame, images_lv1_ori), 1)

                images_lv2_ori = Variable(blur_image_s2 - 0.5).unsqueeze(0).cuda()
                pre_frame_s2 = F.interpolate(pre_deblur_frame, scale_factor=0.5, mode='bilinear')
                images_lv2 = torch.cat((pre_frame_s2, images_lv2_ori), 1)

                images_lv3_ori = Variable(blur_image_s3 - 0.5).unsqueeze(0).cuda()
                pre_frame_s3 = F.interpolate(pre_deblur_frame, scale_factor=0.25, mode='bilinear')
                images_lv3 = torch.cat((pre_frame_s3, images_lv3_ori), 1)

                images_lv4_ori = Variable(blur_image_s4 - 0.5).unsqueeze(0).cuda()
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

                start = time.time()
                #Lv4
                images_lv4_com = torch.cat((images_lv4_1, images_lv4_2, images_lv4_3, images_lv4_4, images_lv4_5, images_lv4_6, images_lv4_7, images_lv4_8), 0)
                feature_lv4_com, feature_lv4_x1, feature_lv4_x2 = encoder_lv4(images_lv4_com)
                feature_lv4_1 = feature_lv4_com[:1,:,:,:]
                feature_lv4_2 = feature_lv4_com[1:2,:,:,:]
                feature_lv4_3 = feature_lv4_com[2:3,:,:,:]
                feature_lv4_4 = feature_lv4_com[3:4,:,:,:]
                feature_lv4_5 = feature_lv4_com[4:5,:,:,:]
                feature_lv4_6 = feature_lv4_com[5:6,:,:,:]
                feature_lv4_7 = feature_lv4_com[6:7,:,:,:]
                feature_lv4_8 = feature_lv4_com[7:,:,:,:]
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

                #Lv3
                images_lv3_1_res = residual_lv4_de[:1,:,:,:] + images_lv3_1
                images_lv3_2_res = residual_lv4_de[1:2,:,:,:] + images_lv3_2
                images_lv3_3_res = residual_lv4_de[2:3,:,:,:] + images_lv3_3
                images_lv3_4_res = residual_lv4_de[3:,:,:,:] + images_lv3_4
                images_lv3_com = torch.cat((images_lv3_1_res, images_lv3_2_res, images_lv3_3_res, images_lv3_4_res), 0)
                feature_lv3_com, feature_lv3_x1, feature_lv3_x2 = encoder_lv3(images_lv3_com)
                feature_lv3_1 = feature_lv3_com[:1,:,:,:]
                feature_lv3_2 = feature_lv3_com[1:2,:,:,:]
                feature_lv3_3 = feature_lv3_com[2:3,:,:,:]
                feature_lv3_4 = feature_lv3_com[3:,:,:,:]
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

                #Lv2
                images_lv2_1_res = residual_lv3_de[:1,:,:,:] + images_lv2_1
                images_lv2_2_res = residual_lv3_de[1:,:,:,:] + images_lv2_2
                images_lv2_com = torch.cat((images_lv2_1_res, images_lv2_2_res), 0)
                feature_lv2_com,  feature_lv2_x1, feature_lv2_x2= encoder_lv2(images_lv2_com)
                feature_lv2_1 = feature_lv2_com[:1,:,:,:]
                feature_lv2_2 = feature_lv2_com[1:,:,:,:]
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

                #Lv1
                feature_lv1, feature_lv1_x1, feature_lv1_x2 = encoder_lv1(images_lv1 + residual_lv2)
                feature_lv1_x1 = F.interpolate(feature_lv1_x1, scale_factor=2, mode='bilinear')
                deblur_image = decoder_lv1(feature_lv1 + feature_lv2, feature_lv1_x1, feature_lv1_x2)
                duration = time.time()-start

                image_deblur = F.interpolate(deblur_image, (H_ori, W_ori), mode='bilinear')
                pre_deblur_frame.data = image_deblur.data
                
                test_time += duration
                print('%s %s/%s RunTime:%.4f'%(DATASET, video_name, frames_list[i], duration), '  Average Runtime:%.4f'%(test_time/(iteration+1)))
                save_images(image_deblur.data + 0.5, video_name, frames_list[i]) 
                iteration += 1

        print('Video: %s, count: %d, Average time: %.4f'%(video_name, len(frames_list), (test_time/iteration)))

    print('****************Average*********************')
    print('%s: Average time: %.4f'%(EXPDIR, test_time/iteration))
            
if __name__ == '__main__':
    main()

        

        

