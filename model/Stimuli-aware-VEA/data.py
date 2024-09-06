# This code is written by Jingyuan Yang @ XD

"""FI_8 Dataset class"""

from __future__ import absolute_import
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils import data
import os
import torch

class FI_8(data.Dataset):
    """Flickr & Instagram 8-classification dataset
        Args:
            csv_file: a 3-column csv_file
                      column 1 contains the names of images
                      column 2 and 3 represents the emo label and senti label
            root_dir: directory to the images
            transform: preprocessing and augmentation of the images
    """

    def __init__(self, csv_file, root_dir, face_dir,  rcnn_dir, transform, face_transform):
        self.annotations = pd.read_csv(csv_file, header=None) # header = None: means no header
                                                              # header = 0: means the first row is header
        self.root_dir = root_dir
        self.face_dir = face_dir
        # self.sal_dir = sal_dir
        self.rcnn_dir = rcnn_dir
        self.transform = transform
        self.face_transform = face_transform
        # self.sal_transform = sal_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = str(self.root_dir) + str(self.annotations.iloc[idx, 0]).split('.')[0] + '.jpg' # 0 1
        image = Image.open(img_name)
        image = image.convert("RGB")

        # sal_name = str(self.sal_dir) + str(self.annotations.iloc[idx, 0]).split('.')[0] + '_sal_fuse.png'  # 0 1
        # saliency = Image.open(sal_name)
        # saliency = saliency.convert("RGB")

        label_emo = self.annotations.iloc[idx, 1]
        label_senti = self.annotations.iloc[idx, 2] # emotion label [0,7] # 1 3

        image = self.transform(image)
        # saliency = self.sal_transform(saliency)

        face_name = str(self.face_dir) + str(self.annotations.iloc[idx, 0]).split('.')[0] + '.jpg'
        if os.path.exists(face_name):
            # face=Image.open(face_name).convert('L')
            face = Image.open(face_name)
            face = face.convert("RGB")
            face = self.face_transform(face)
            fmask=torch.ones(1)
        else:
            face = torch.zeros(3, 44, 44)
            fmask=torch.zeros(1)

        rcnn_name = str(self.rcnn_dir) + str(self.annotations.iloc[idx, 0]).split('.')[0] + '.npy'
        rcnn = np.load(rcnn_name)[:18,:]#.reshape(-1)
        # rcnn = np.load(rcnn_name)
        rcnn = torch.from_numpy(rcnn.astype(np.float32))
        # print(rcnn.size())

        sample = {'img_id': str(self.annotations.iloc[idx, 0]).split('.')[0] + '.jpg', 'image': image, 'label_senti': label_senti, 'label_emo': label_emo,
                  'face': face,  'rcnn':rcnn, 'face_mask':fmask}

        return sample
