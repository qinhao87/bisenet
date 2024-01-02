# import sys
# sys.path.append(r'F:\py_pro\BiSeNet')
import torch
import glob
import os
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from imgaug import augmenters as iaa
import imgaug as ia
from utils import get_label_info, one_hot_it, RandomCrop, reverse_one_hot, one_hot_it_v11, one_hot_it_v11_dice
import random

def augmentation():
    # augment images with spatial transformation: Flip, Affine, Rotation, etc...
    # see https://github.com/aleju/imgaug for more details
    pass


def augmentation_pixel():
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    pass

class crack(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, loss='dice', mode='train'):
        super().__init__()
        # import ipdb;ipdb.set_trace()
        self.mode = mode
        self.image_list = [os.path.join(image_path,image_name) for image_name in os.listdir(image_path)]

        self.image_list.sort()
        self.label_list = [os.path.join(label_path,label_name) for label_name in os.listdir(label_path)]
        self.label_list.sort()

        self.T = A.Compose([
            A.PadIfNeeded(min_height=600,min_width=600,
                          border_mode=cv2.BORDER_CONSTANT,value=0),
                          ToTensorV2(),
        ])

        self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]
        self.loss = loss

    def __getitem__(self, index):
        # load image and crop
        # import ipdb;ipdb.set_trace()
        img = cv2.imread(self.image_list[index],cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(self.label_list[index],cv2.IMREAD_GRAYSCALE)

        # if self.mode == 'train':
        h,w = img.shape
        if h>600 or w>600:
            # import ipdb;ipdb.set_trace()
            resize_t = A.LongestMaxSize(max_size=600)
            img,label = resize_t(image=img,mask=label)['image'],resize_t(image=img,mask=label)['mask']
            label[label>0] = 1
        transformed = self.T(image=img,mask=label)
        img,label = transformed['image'] / 255.0,transformed['mask']
            

        if self.loss == 'dice':
            # label -> [num_classes, H, W]
            # import ipdb;ipdb.set_trace()
            not_label = torch.logical_not(label)
            label = torch.stack([not_label,label],dim=0)

            return img, label.long()

        elif self.loss == 'crossentropy':
            # label = one_hot_it_v11(label, self.label_info).astype(np.uint8)
            # label = label.astype(np.float32)
            # label = torch.from_numpy(label).long()

            return img, label.long()

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    # import ipdb;ipdb.set_trace()
    # data = CamVid('/path/to/CamVid/train', '/path/to/CamVid/train_labels', '/path/to/CamVid/class_dict.csv', (640, 640))
    data = crack(r'F:/py_pro/BiSeNet/img/train1', r'F:/py_pro/BiSeNet/img/label1',
                  loss='crossentropy', mode='val')
    from model.build_BiSeNet import BiSeNet
    from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy

    # label_info = get_label_info('/data/sqy/CamVid/class_dict.csv')
    for i, (img, label) in enumerate(data):
        
        print(label.size())
        print(torch.max(label))

