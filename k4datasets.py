import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from skimage import io, transform
import os
import math
import numpy as np
import random

class K4Dataset(Dataset):
    def __init__(self, blur_image_files, sharp_image_files, root_dir, crop=False, crop_size=1536, multi_scale=False, rotation=False, color_augment=False, transform=None):
        """
        Args:
             split_file: Path to the split file
             root_dir: Directory with all the images
             transform: Optional transform to be appeared on a sample
        """
        blur_file = open(blur_image_files, 'r')
        self.blur_image_files = blur_file.readlines()
        sharp_file = open(sharp_image_files, 'r')
        self.sharp_image_files = sharp_file.readlines()
        self.root_dir = root_dir
        self.transform = transform        
        self.crop = crop
        self.crop_size = crop_size
        self.multi_scale = multi_scale
        self.rotation = rotation
        self.color_augment = color_augment
        self.rotate90 = transforms.RandomRotation(90)  
        self.rotate45 = transforms.RandomRotation(45)    

    def __len__(self):
        return len(self.blur_image_files)

    def __getitem__(self, idx):
        image_name = self.blur_image_files[idx][0:-1].split('/')
        image_num = int(image_name[3].split('.')[0])
        # detect the head of a video file
        if not os.path.exists(os.path.join(self.root_dir, image_name[0], image_name[1], image_name[2], '%05d.jpg'%(image_num-1))):
            video_start = True
        else:
            video_start = False

        original_image = Image.open(os.path.join(self.root_dir, image_name[0], image_name[1], image_name[2], image_name[3])).convert('RGB')
        W, H = original_image.size
        temp_image = original_image.resize((int(W/2), int(H/2)), Image.ANTIALIAS) # down-sample blur image to half size
        blur_image = temp_image.resize((W, H), Image.BILINEAR) # up-sample blur image to original size
        sharp_image = Image.open(os.path.join(self.root_dir, image_name[0], image_name[1], 'sharp', image_name[3])).convert('RGB')
        
        if self.rotation:
            degree = random.choice([90, 180, 270])
            blur_image = transforms.functional.rotate(blur_image, degree) 
            sharp_image = transforms.functional.rotate(sharp_image, degree)

        if self.color_augment:
            #contrast_factor = 1 + (0.2 - 0.4*np.random.rand())
            #blur_image = transforms.functional.adjust_contrast(blur_image, contrast_factor)
            #sharp_image = transforms.functional.adjust_contrast(sharp_image, contrast_factor)
            blur_image = transforms.functional.adjust_gamma(blur_image, 1)
            sharp_image = transforms.functional.adjust_gamma(sharp_image, 1)                           
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            blur_image = transforms.functional.adjust_saturation(blur_image, sat_factor)
            sharp_image = transforms.functional.adjust_saturation(sharp_image, sat_factor)
            
        if self.transform:
            blur_image = self.transform(blur_image)
            # blur_image = blur_image
            sharp_image = self.transform(sharp_image)

        if self.crop:
            W = blur_image.size()[1]
            H = blur_image.size()[2]

            if video_start:
                self.Ws = np.random.randint(0, W-self.crop_size-1, 1)[0]
                self.Hs = np.random.randint(0, H-self.crop_size-1, 1)[0]
            # print('Ws: %d, Hs: %d'%(self.Ws, self.Hs))
            
            blur_image = blur_image[:, self.Ws:self.Ws+self.crop_size, self.Hs:self.Hs+self.crop_size]
            sharp_image = sharp_image[:, self.Ws:self.Ws+self.crop_size, self.Hs:self.Hs+self.crop_size]
                       
        if self.multi_scale:
            if self.crop:
                H = sharp_image.size()[1]
                W = sharp_image.size()[2]
            else:
                H = int(math.ceil(sharp_image.size()[1]/128)*128)
                W = int(math.ceil(sharp_image.size()[2]/128)*128)
            blur_image_s1 = transforms.ToPILImage()(blur_image)
            sharp_image_s1 = transforms.ToPILImage()(sharp_image)
            # blur_image_s1 = blur_image
            # sharp_image_s1 = sharp_image
            # blur_image_s2 = transforms.Resize([H/2, W/2])(blur_image_s1)
            # sharp_image_s2 = transforms.Resize([H/2, W/2])(sharp_image_s1)
            # blur_image_s3 = transforms.Resize([H/4, W/4])(blur_image_s1)
            # sharp_image_s3 = transforms.Resize([H/4, W/4])(sharp_image_s1)

            blur_image_s1 = transforms.Resize([int(H), int(W)])(blur_image_s1)
            sharp_image_s1 = transforms.Resize([int(H/2), int(W/2)])(sharp_image_s1)
            blur_image_s2 = transforms.ToTensor()(transforms.Resize([int(H/2), int(W/2)])(blur_image_s1))
            sharp_image_s2 = transforms.ToTensor()(transforms.Resize([int(H/4), int(W/4)])(sharp_image_s1))
            blur_image_s3 = transforms.ToTensor()(transforms.Resize([int(H/4), int(W/4)])(blur_image_s1))
            sharp_image_s3 = transforms.ToTensor()(transforms.Resize([int(H/8), int(W/8)])(sharp_image_s1))
            blur_image_s4 = transforms.ToTensor()(transforms.Resize([int(H/8), int(W/8)])(blur_image_s1))

            blur_image_s1 = transforms.ToTensor()(blur_image_s1)
            sharp_image_s1 = transforms.ToTensor()(sharp_image_s1)
            return {'blur_image_s1': blur_image_s1,
                    'blur_image_s2': blur_image_s2,
                    'blur_image_s3': blur_image_s3,
                    'blur_image_s4': blur_image_s4,
                    'sharp_image': sharp_image,
                    'sharp_image_s1': sharp_image_s1,
                    'sharp_image_s2': sharp_image_s2,
                    'sharp_image_s3': sharp_image_s3,
                    'video_start': video_start}
            # return {'blur_image_s1': blur_image_s1, 'blur_image_s2': blur_image_s2, 'blur_image_s3': blur_image_s3, 'sharp_image_s1': sharp_image_s1, 'sharp_image_s2': sharp_image_s2, 'sharp_image_s3': sharp_image_s3}
        else:
            return {'blur_image': blur_image,
                    'sharp_image': sharp_image,
                    'video_start': video_start} # add video_start
        
