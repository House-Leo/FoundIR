import os
from os import path as osp
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset 
from PIL import Image
from pathlib import Path
import numpy as np
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import Augmentor
import math
import torch
import cv2

def paired_paths_from_meta_info_file(folders, keys, meta_info_file):

    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, f'The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}'
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with open(meta_info_file, 'r') as fin:
        # gt_names = [line.strip().split(' ')[0] for line in fin]
        paths = [line.strip() for line in fin]

    # paths = []
    p = []
    for path in paths:
        gt_path, lq_path = path.split(', ')
        gt_path = osp.join(gt_folder, gt_path)
        lq_path = osp.join(input_folder, lq_path)
        p.append(dict([(f'{input_key}_path', lq_path), (f'{gt_key}_path', gt_path)]))
    return p

class CombinedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.equalizeHist = equalizeHist
        self.augment_flip = augment_flip
        self.crop_patch = crop_patch
        self.generation = generation
        self.image_size = image_size
        self.opt = opt
        self.task = task
        #origin----------------------------------------------------------------------------------------------------------
        if task == None:
            self.dir_LQ = os.path.join(opt.dataroot, 'LQ')
            self.dir_GT = os.path.join(opt.dataroot, 'GT')

            self.A_paths = sorted(make_dataset(self.dir_LQ, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_GT, opt.max_dataset_size))

        if task == 'meta_info' and opt.meta is not None:
            self.dir_LQ = opt.dataroot
            self.dir_GT = opt.dataroot

            self.paths = paired_paths_from_meta_info_file(
                [self.dir_LQ, self.dir_GT], ['adap', 'gt'], opt.meta)
            
            self.A_paths = [path['adap_path'] for path in self.paths]
            self.B_paths = [path['gt_path'] for path in self.paths]
            self.A_size = len(self.A_paths)
            print(len(self.A_paths))
            self.B_size = len(self.B_paths)
            print(len(self.B_paths))

        else:
            self.A_size = len(self.A_paths)  # get the size of dataset A
            print(self.A_size)
            self.B_size = len(self.B_paths)  # get the size of dataset B
            print(self.B_size)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        if self.task == 'meta_info' and self.opt.meta is not None:
            paths = self.paths[index]
            A_path = paths['adap_path']
            B_path = paths['gt_path']
        else:
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            B_path = self.B_paths[index % self.B_size]

        condition = Image.open(A_path).convert('RGB') #condition
        gt = Image.open(B_path).convert('RGB') #gt
        
        w, h = condition.size
        transform_params = get_params(self.opt, condition.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=False)
        B_transform = get_transform(self.opt, transform_params, grayscale=False)
        condition = A_transform(condition)
        gt = B_transform(gt)
        if self.opt.phase == 'train':
            if h < 256 or w < 256:
                osize = [256, 256]
                resi = transforms.Resize(osize, transforms.InterpolationMode.BICUBIC)
                condition = resi(condition)
                gt = resi(gt)
        
        # if self.opt.phase == 'test':
        #     if h > 1024 or w > 1024:


                
        return {'adap': condition, 'gt': gt, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)
    
    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                return [p for ext in self.exts for p in Path(f'{flist}').glob(f'**/*.{ext}')]

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []

    def cv2equalizeHist(self, img):
        (b, g, r) = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img = cv2.merge((b, g, r))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = TF.to_tensor(img).float()
        return img_t

    def load_name(self, index, sub_dir=False):
        if self.condition:
            # condition
            name = self.input[index]
            if sub_dir == 0:
                return os.path.basename(name)
            elif sub_dir == 1:
                path = os.path.dirname(name)
                sub_dir = (path.split("/"))[-1]
                return sub_dir+"_"+os.path.basename(name)

    def get_patch(self, image_list, patch_size):
        i = 0
        h, w = image_list[0].shape[:2]
        rr = random.randint(0, h-patch_size)
        cc = random.randint(0, w-patch_size)
        for img in image_list:
            image_list[i] = img[rr:rr+patch_size, cc:cc+patch_size, :]
            i += 1
        return image_list

    def pad_img(self, img_list, patch_size, block_size=8):
        i = 0
        for img in img_list:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            bottom = 0
            right = 0
            if h < patch_size:
                bottom = patch_size-h
                h = patch_size
            if w < patch_size:
                right = patch_size-w
                w = patch_size
            bottom = bottom + (h // block_size) * block_size + \
                (block_size if h % block_size != 0 else 0) - h
            right = right + (w // block_size) * block_size + \
                (block_size if w % block_size != 0 else 0) - w
            img_list[i] = cv2.copyMakeBorder(
                img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            i += 1
        return img_list
    
    def get_pad_size(self, index, block_size=8):
        img = Image.open(self.input[index])
        patch_size = self.image_size
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        bottom = 0
        right = 0
        if h < patch_size:
            bottom = patch_size-h
            h = patch_size
        if w < patch_size:
            right = patch_size-w
            w = patch_size
        bottom = bottom + (h // block_size) * block_size + \
            (block_size if h % block_size != 0 else 0) - h
        right = right + (w // block_size) * block_size + \
            (block_size if w % block_size != 0 else 0) - w
        return [bottom, right]

    def split_image(image, patch_size, overlap):
        w, h = image.size
        patches = []
        positions = []

        for i in range(0, w, patch_size - overlap):
            for j in range(0, h, patch_size - overlap):
                right = min(i + patch_size, w)
                bottom = min(j + patch_size, h)
                patch = image.crop((i, j, right, bottom))
                patches.append(np.array(patch))
                positions.append((i, j))
        return patches, positions

    def merge_patches(patches, positions, image_size, patch_size, overlap):
        w, h = image_size
        result = np.zeros((h, w, 3), dtype=np.float32)
        weight = np.zeros((h, w, 3), dtype=np.float32)

        for patch, (i, j) in zip(patches, positions):
            patch_h, patch_w = patch.shape[:2]
            result[j:j+patch_h, i:i+patch_w] += patch
            weight[j:j+patch_h, i:i+patch_w] += 1
        result /= weight
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))