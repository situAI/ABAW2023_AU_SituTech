
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import logging
import time
from PIL import Image
import numpy as np
import glob
import random

from core.utils import DATASET_REGISTRY

# Parameters
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
VID_FORMATS = ["mp4", "mov", "avi", "mkv"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])
VID_FORMATS.extend([f.upper() for f in VID_FORMATS])


@DATASET_REGISTRY.register()
class ImageDataset(Dataset):
    def __init__(self, image_root, label_root, paral_image_roots=[], paral_drop=0.3, class_num=12, image_size=(112, 112), rank=-1, task="train"):

        t0 = time.time()
        self.__dict__.update(locals())
        self.task = self.task.lower()
        self.local_rank = torch.distributed.get_rank()
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.class_num_list = [0 for _ in range(self.class_num)]
        self.__check_file_exists()
        
        # transformer
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.6, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize(self.image_size[0]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            
        ])

        # load image path
        img_id_list, img_label_list = self.load_dataset(image_root, label_root, paral=False)
        if self.local_rank == 0:
            logging.info(f'{self.task} init class num list: {self.class_num_list}')
        

        if self.task in ['train']:
            for paral_image_root in paral_image_roots:
                _img_id_list, _img_label_list = self.load_dataset(paral_image_root, label_root, paral=True)
                img_id_list += _img_id_list
                img_label_list += _img_label_list

        self.img_id_list, self.img_label_list = img_id_list, img_label_list

        t1 = time.time()
        if self.local_rank == 0:
            loss_w_list = [max(self.class_num_list) / c for c in self.class_num_list]
            logging.info(f'{self.task} loss weight: {loss_w_list}')
            logging.info(f'{self.task} end class num list: {self.class_num_list}')
            logging.info(f"%.1fs for {self.task} dataset initialization." % (t1 - t0))
    
    def __check_file_exists(self):
        check_file_list =  [self.image_root, self.label_root] + self.paral_image_roots
        for file in check_file_list:
            if not os.path.exists(file):
                logging.info(f'file is not exist. {file}')
        return
    
    def __init_class_weight(self):
        class_avg_num = sum(self.class_num_list) / self.class_num
        self.class_weight_list = [(c/class_avg_num) for c in self.class_num_list]


    def data_balancing(self, img_ids, img_labels):
        # 数据均衡
        self.__init_class_weight()
        std_id_list, std_label_list = list(), list()
        for img_id, img_label in zip(img_ids, img_labels):
            drop_code = 0
            for l_idx, l in enumerate(img_label):
                if int(l) == 1:
                    if self.class_weight_list[l_idx] < 1:
                        drop_code = 0
                        break
                    elif random.random() < self.class_weight_list[l_idx] / max(self.class_weight_list): 
                        drop_code = 1
                    
            if drop_code == 1:
                self.class_num_list= [self.class_num_list[l_idx] - 1 if int(l) == 1 else self.class_num_list[l_idx] for l_idx, l in enumerate(img_label)]
            else:
                std_id_list.append(img_id)
                std_label_list.append(img_label)
        
        return std_id_list, std_label_list


    def get_img_label_ds(self, img_root, label_root, paral=False):
        logging.info(f'load image root... {img_root}')
        img_ids_list, img_labels_list  = list(), list()

        for txt_path in glob.glob(os.path.join(label_root, '*.txt')):
            video_name = os.path.splitext(txt_path.split('/')[-1])[0]
            image_dir = os.path.join(img_root, video_name)
            img_ids, img_labels = list(), list()
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[1:]):
                    image_path = os.path.join(image_dir, f'{i+1 :05}.jpg')
                    value_list = [int(value) for value in line.strip('\n').split(',')]
                    if (-1 in value_list and 1 not in value_list) or not os.path.exists(image_path):
                        continue
                    if -1 in value_list:
                        value_list = [0 if i == -1 else i for i in value_list]
                    if paral:
                        if random.random() < self.paral_drop:
                            continue
                        if 'upcover' in img_root:
                            value_list = [0,0,0,0,0] + value_list[5:] 
                        elif 'downcover' in img_root:
                            value_list =  value_list[:5] + [0,0,0,0,0,0,0]
                        
                    img_ids.append(image_path)
                    img_labels.append(value_list)
                    self.class_num_list = [self.class_num_list[vi]+1 if v == 1 else self.class_num_list[vi] for vi, v in enumerate(value_list)]
            
            img_ids_list.append(img_ids)
            img_labels_list.append(img_labels)

        return img_ids_list, img_labels_list 


    def load_dataset(self, img_root, label_root, paral=False):

        std_id_list, std_label_list = list(), list()
        img_ids_list, img_labels_list = self.get_img_label_ds(img_root, label_root, paral)
        for img_ids, img_labels in zip(img_ids_list, img_labels_list):
            if self.task == 'train':
                img_ids, img_labels = self.data_balancing(img_ids, img_labels)
            std_id_list += img_ids
            std_label_list += img_labels
    

        return std_id_list, std_label_list

    def __len__(self):
        return len(self.img_id_list)

    def __getitem__(self, idx):

        img_path, label = self.self.img_id_list[idx], self.img_label_list[idx]
        # img = cv2.imread(img_path)
        img = Image.open(img_path)
        if self.task == 'train':
            data = self.train_transforms(img)
        elif self.task == 'valid' or self.task == 'test':
            data = self.val_transforms(img)
        else:
            logging.info(f"ERROR DATA TYPE {self.task}")

        return data, np.asarray(label)


@DATASET_REGISTRY.register()
class ABAW5(Dataset):
    def __init__(self, data_root, image_size, caption):

        self.img_list = list()
        self.__dict__.update(locals())
    
        self.transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.3, contrast=0.3,
            #                       saturation=0.3, hue=0.3),
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        for root, dirs, files in os.walk(data_root):
            for name in files:
                if name[-1] == 'g':
                    self.img_list.append(os.path.join(root, name))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        video_name = img_path.split('/')[-2]
        img_name = img_path.split('/')[-1].split('.')[0]
        name = f'{self.caption}' + video_name + '/' + img_name
        img = Image.open(img_path)
        data = self.transform(img)

        return data, name