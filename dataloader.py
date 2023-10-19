import os
from scipy import io as sio
import torch
from torch.utils import data
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import random

def random_crop(im_h, im_w, crop_h, crop_w):
        res_h = im_h - crop_h
        res_w = im_w - crop_w
        i = random.randint(0, res_h)
        j = random.randint(0, res_w)
        return i, j, crop_h, crop_w

class SHHA_loader(data.Dataset):
    def __init__(self, data_path, split, crop_size=512):
        self.data_path = os.path.join(data_path, split + "_data")
        self.img_path = os.path.join(self.data_path, "images")
        self.gt_path = os.path.join(self.data_path, "ground_truth")

        self.file_names = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path, filename))]
        
        self.crop_size = crop_size

        self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __getitem__(self, index):
        image_name = self.file_names[index]
        gt_name = "GT_" + image_name.replace('.jpg', '.mat')
        image = Image.open(os.path.join(self.img_path, image_name)).convert('RGB')
        gt = sio.loadmat(os.path.join(self.gt_path, gt_name))
        gt = gt["image_info"][0, 0][0, 0][0] - 1
        return self.train_transform(image, gt)

    def __len__(self):
        return len(self.file_names)
            
    def train_transform(self, img, keypoints):
        """random crop image patch and find people in it"""
        wd, ht = img.size
        st_size = min(wd, ht)
        assert st_size >= self.crop_size
        assert len(keypoints) > 0
        i, j, h, w = random_crop(ht, wd, self.crop_size, self.crop_size)
        img = F.crop(img, i, j, h, w)

        mask = (keypoints[:, 0] >= j) & (keypoints[:, 0] < j + w) & \
               (keypoints[:, 1] >= i) & (keypoints[:, 1] < i + h)
        keypoints = keypoints[mask]
        keypoints[:, 0] = keypoints[:, 0] - j
        keypoints[:, 1] = keypoints[:, 1] - i

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float()