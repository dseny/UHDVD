# Multi-Scale Separable Network for Ultra-High-Defnition Video Deblurring
Pytorch Implementation of ICCV21 "[Multi-Scale Separable Network for Ultra-High-Defnition Video Deblurring]" <br/>

![Pipeline of UHDVD](./figures/framework.png)

## 4KRD
![Exampls of 4KRD](./figures/4krd.png)
Download 4KRD dataset: https://drive.google.com/drive/folders/19bjJLMgQkwIAQaZYvsUhEVaxzJQFwhHF?usp=sharing
Please download training datasets (4KRD/GoPro/DVD/REDS) into './datas/XXXX'. <br/>
Running the following command to obtain files: 'train_blur_list.txt' and 'train_sharp_list.txt'
```
python txt_list.py
```

## Dependences
4KRD Pretrained models are stored in './checkpoints/4KRD'. 

__For requires, run following commands.__
```
pip install -r requirements.txt
```

## Running
__For training, run following commands.__

```
python train_video.py -se start_epoch -g GPU_number
```

__For testing, put test samples as './test_set/XXXX/XXXX/blurry', then run following commands.__

```
python test_video.py -g GPU_number
```
The results will be saved at './test_set/XXXX/XXXX/deblur_4KRD'.

## Citation
If you think this work is useful for your research, please cite the following paper.

```
@InProceedings{Deng_ICCV21,
author = {Deng, Senyou and Ren, Wenqi and Yan, Yanyang and Wang, Tao and Song, Fenglong and Cao, Xiaochun},
title = {Multi-Scale Separable Network for Ultra-High-Defnition Video Deblurring},
booktitle = {International Conference on Computer Vision (ICCV)},
month = {October},
year = {2021}
}
```
