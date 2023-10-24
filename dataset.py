import os
import numpy as np
import pandas as pd

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.ops import box_convert
import cv2
import torchvision.transforms as T
from PIL import Image

from util.box_ops import box_xyxy_to_cxcywh

def get_transforms():

	transform = T.Compose([
    T.Resize((800, 1333)),
    T.ToTensor()
])

	return transform

class MTSD_Dataset(Dataset):
    
    def __init__(self, dataset_dir, label_dir, transforms=None):

        self.dataset_dir = dataset_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.image_names = [file for file in sorted(os.listdir(os.path.join(dataset_dir))) if file.endswith('.jpg')]

    def __getitem__(self, index):

        image_path = os.path.join(self.dataset_dir, self.image_names[index])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        ### Load label file ###
        label_path = os.path.join(self.label_dir, self.image_names[index][:-3]+"txt")
        with open(label_path, 'r') as f:
            label_f = f.readlines()
            label_f = [label.split() for label in label_f]
        
        ### add attributes to label dict
        target = {}
        boxes, areas, labels = [], [], []
        for label in label_f:
            class_id, xmin, ymin, xmax, ymax = int(label[0]), float(label[1]), float(label[2]), float(label[3]), float(label[4])
            cls = class_id
            area = xmin * ymin
            # xmin, xmax, ymin, ymax = x-w, x+w, y-h, y+h
            box = [xmin, ymin, xmax, ymax]
            # box = box_xyxy_to_cxcywh(box)
            # w, h = box[2], box[3]
            # box = box / torch.tensor([w, h, w, h])
            # print("CONVERT: ", box)
            boxes.append(box)
            labels.append(class_id)
            areas.append(area)
        
        boxes = torch.tensor(boxes)
        # boxes = box_xyxy_to_cxcywh(boxes)
        
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        target["image_id"] = torch.tensor([index])
        target["boxes"] = boxes
        target["areas"] = torch.tensor(areas)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["iscrowd"] = torch.tensor(labels, dtype=torch.int64)
        target["orig_size"] = torch.tensor([image.size], dtype=torch.int64)

        if self.transforms != None:
            image = self.transforms(image)
        return image, target

    def __len__(self):
        return len(self.image_names)
    
# dataset = MTSD_Dataset(dataset_dir = "/home/harry/backdoor/BLUE_LOW_SCALING/images/train", label_dir = "/home/harry/backdoor/BLUE_LOW_SCALING/labels/train",
#                         transforms = get_transforms())
