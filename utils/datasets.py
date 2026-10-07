'''
FSGPNet、SCTransNet等大多数的dataset
'''
import os.path as osp
from utils.images import load_image
import matplotlib.pyplot as plt
import os
from .utils2 import *
import cv2
import numpy as np
import torch
import random
from torch.utils.data.dataset import Dataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


class NUDTSIRSTSetLoader(Dataset):
    def __init__(self, base_dir='../data/NUDT-SIRST/', mode='test', img_norm_cfg=None, patchSize=256):
        super(NUDTSIRSTSetLoader).__init__()
        self.mode = mode

        if mode == 'trainval':
            txtfile = 'train_NUDT-SIRST.txt'
        elif mode == 'test':
            txtfile = 'test_NUDT-SIRST.txt'
        else:
            raise NotImplementedError

        self.list_dir = osp.join(base_dir, 'img_idx', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg('NUDT-SIRST', base_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        if mode == 'trainval':
            self.tranform = augumentation()
        self.patch_size = patchSize

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + '.png')

        img = load_image(img_path)
        mask = load_image(label_path)
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        # A few source masks contain antialiased boundary pixels. Convert to
        # the same binary ground truth for training and every evaluation metric.
        mask = (np.array(mask, dtype=np.float32) > 127.5).astype(np.float32)
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        # h, w = img.shape # All the pictures are 512 x 512
        if self.mode == 'trainval':

            img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)
            img_patch, mask_patch = self.tranform(img_patch, mask_patch)
            img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]
            img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
            mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        else:
            h, w = img.shape
            img = PadImg(img)
            mask = PadImg(mask)

            img, mask = img[np.newaxis, :], mask[np.newaxis, :]

            img = torch.from_numpy(np.ascontiguousarray(img))
            mask = torch.from_numpy(np.ascontiguousarray(mask))

        if self.mode == 'trainval':
            return img_patch, mask_patch
        else:
            # return img, [h,w], self.names[i]
            return img, mask, [h, w], self.names[i]

    def __len__(self):
        return len(self.names)


class IRSTD1KSetLoader(Dataset):
    def __init__(self, base_dir='./data/IRSTD-1K/', mode='test', img_norm_cfg=None, patchSize=512):
        super(IRSTD1KSetLoader).__init__()
        self.mode = mode

        if mode == 'trainval':
            txtfile = 'train_IRSTD-1K.txt'
        elif mode == 'test':
            txtfile = 'test_IRSTD-1K.txt'
        else:
            raise NotImplementedError

        self.list_dir = osp.join(base_dir, 'img_idx', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg('IRSTD-1K', base_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
        self.patch_size = patchSize

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + '.png')

        img = load_image(img_path)
        mask = load_image(label_path)

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = (np.array(mask, dtype=np.float32) > 127.5).astype(np.float32)
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        # h, w = img.shape # All the pictures are 512 x 512
        if self.mode == 'trainval':

            img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)
            img_patch, mask_patch = self.tranform(img_patch, mask_patch)
            img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]
            img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
            mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        else:
            h, w = img.shape
            img = PadImg(img)
            mask = PadImg(mask)

            img, mask = img[np.newaxis, :], mask[np.newaxis, :]

            img = torch.from_numpy(np.ascontiguousarray(img))
            mask = torch.from_numpy(np.ascontiguousarray(mask))

        if self.mode == 'trainval':
            return img_patch, mask_patch
        else:
            # return img, [h,w], self.names[i]
            return img, mask, [h, w], self.names[i]

    def __len__(self):
        return len(self.names)


class augumentation(object):
    def __call__(self, input, target):
        if random.random() < 0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target
