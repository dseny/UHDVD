from __future__ import division
import os
import glob

root_dir = './train/'

file_list = os.listdir(root_dir)
print(file_list)

fp_blur = open("./train_blur_list.txt","w")
fp_sharp = open("./train_sharp_list.txt","w")
for file in file_list:
    frames = glob.glob(os.path.join(root_dir, file) + '/sharp/*.png')
    frames = sorted(frames)
    for frame in frames:
        content = frame.replace('./', '')
        fp_sharp.write(content + '\n')
        fp_blur.write(content.replace('sharp', 'blur') + '\n')
fp_blur.close()
fp_sharp.close()
print('Done!')