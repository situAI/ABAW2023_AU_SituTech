import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
import time
import logging
import glob
import random

from core.utils import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class FeatSeqDataset(Dataset):
    def __init__(self,
                feat_root,
                label_root,
                img_root,
                seq_len,
                feat_list,
                audio_feat_list = ["ecapatdnn", "hubert", "wav2vec", "wav2vec_emotion", "fbank"],
                class_num = 12,
                class_num_list = [],
                task='',
                feat_map={}
                ):
        """FeatSeqDataset: 根据label_root获取对应的数据集

        Args:
            feat_root (str): 特征h5文件存放目录
            label_root (str): 标签txt文件存放目录
            seq_len (int): 特征序列长度
            feat_list (list): 需要输入模型的特征文件名 不包括.h5后缀 一维list
            audio_feat_list(opt, list): 输入特征中为音频特征的list
            task(opt, str): 数据集任务: train valid test
            feat_map(init, dict): {key:特征名 value:h5特征}
        """

        t0 = time.time()
        self.__dict__.update(locals())
        self.local_rank = torch.distributed.get_rank()
        self.class_num_list = [0 for _ in range(self.class_num)]
        self.feat_list = [list(_feat.keys())[0] if isinstance(_feat,dict) else _feat for _feat in self.feat_list]
        self.__check_file_exists()
        self.seqfeat_id_list, self.seqfeat_label_list = self.get_seqfeat_data()
        t1 = time.time()
        if self.local_rank == 0:
            loss_w_list = [max(self.class_num_list) / c for c in self.class_num_list]
            logging.info(f'{self.task} loss weight list: {loss_w_list}. class num list: {self.class_num_list}.')
            logging.info(f" %.1fs for {self.task} dataset initialization." % (t1 - t0))


    def __check_file_exists(self):
        check_file_list =  [self.feat_root, self.label_root]
        for file in check_file_list:
            if not os.path.exists(file):
                logging.info(f'file is not exist. {file}')
        return


    def load_feat_data(self):
        # 获取标签和对应的特征id
        feat_ids_list, feat_labels_list  = list(), list()

        for txt_path in glob.glob(os.path.join(self.label_root,'*.txt')):
            video_name = os.path.splitext(txt_path.split('/')[-1])[0]
            image_dir = os.path.join(self.img_root, video_name)

            with open(txt_path, 'r') as f:
                feat_ids, feat_labels = list(), list()
                lines = f.readlines()
                for i, line in enumerate(lines[1:]):
                    image_path = os.path.join(image_dir, f'{i+1 :05}.jpg')
                    value_list = [float(value) for value in line.strip('\n').split(',')]
                    if -1 in value_list or not os.path.exists(image_path): # 过滤标签
                        continue
                    feat_ids.append(f'{video_name}/{i+1 :05}')   # 生成'video_name/frame_idx'的标签id
                    feat_labels.append(value_list)
                    self.class_num_list = [self.class_num_list[vi]+1 if v == 1 else self.class_num_list[vi] for vi, v in enumerate(value_list)]
            
            feat_ids_list.append(feat_ids)
            feat_labels_list.append(feat_labels)

        return feat_ids_list, feat_labels_list


    def get_seqfeat_data(self):

        seq_id_list, seq_label_list = list(), list()

        feat_ids_list, feat_labels_list = self.load_feat_data()  # 以视频为单位的二维标签list

        seq_len = self.seq_len  # 生成指定序列长度的标签序列
        for feat_id_list, feat_label_list in zip(feat_ids_list, feat_labels_list):

            # 视频有效帧长度小于seq_len进行补充帧
            if len(feat_id_list) < seq_len:  
                feat_id_list = feat_id_list * (seq_len//len(feat_id_list) +1)
                feat_label_list = feat_label_list * (seq_len//len(feat_label_list) +1)

            # 数据序列化
            seq_ids, seq_labels = list(), list()
            for i in range(0, len(feat_id_list), seq_len):
                seq_ids.append(feat_id_list[i: i + seq_len])
                seq_labels.append(feat_label_list[i: i + seq_len])
            seq_ids[-1], seq_labels[-1] = feat_id_list[-seq_len:], feat_label_list[-seq_len:]  # 末尾序列小于seq_len

            seq_id_list += seq_ids
            seq_label_list += seq_labels

        return seq_id_list, seq_label_list


    def open_h5(self, name):
        if isinstance(name, list):
            for n in name:
                self.feat_map[n] = h5py.File(os.path.join(self.feat_root, n + '.h5'), 'r')
        else:
            self.feat_map[name] = h5py.File(os.path.join(self.feat_root, name + '.h5'), 'r')
        return
        

    def close_h5(self, name):
        if isinstance(name, list):
            for n in name:
                self.feat_map[n] = h5py.File(os.path.join(self.feat_root, n + '.h5'), 'r')
        else:
            self.feat_map[name].close()
        return
    
    def get_frame_feats(self, frame_name, feat_list):
        feat = list()
        for feat_name in feat_list:
            # 获取audio特征
            if feat_name in self.audio_feat_list:
                audio_frame_name = frame_name.replace('_right', '').replace('_left', '')
                try:
                    crt_feat = np.asarray(self.feat_map[feat_name][audio_frame_name])
                except:
                    video_name = audio_frame_name.split('/')[0]
                    frame_id = list(self.feat_map[feat_name][video_name].keys())[-1]
                    crt_feat = np.asarray(self.feat_map[feat_name][f'{video_name}/{frame_id}'])
            else: # 获取图像特征
                img_frame_name = frame_name
                try:
                    crt_feat = np.asarray(self.feat_map[feat_name][img_frame_name])
                except:
                    video_name = img_frame_name.split('/')[0]
                    frame_id = list(self.feat_map[feat_name][video_name].keys())[-1]
                    crt_feat = np.asarray(self.feat_map[feat_name][f'{video_name}/{frame_id}'])
            feat.append(crt_feat)
        feat = np.concatenate(feat, axis=-1)

        return feat


    def __len__(self):
        return len(self.seqfeat_id_list)


    def __getitem__(self, idx):
    
        crt_feat_list = self.feat_list            
        seq_list = self.seqfeat_id_list[idx]
        lb_list = self.seqfeat_label_list[idx]

        assert len(seq_list) == len(lb_list) == self.seq_len

        self.open_h5(crt_feat_list)

        seq_feat, seq_label = list(), list()
        for seq_name, label in zip(seq_list, lb_list):
            feat = self.get_frame_feats(seq_name, crt_feat_list)  # 获取seq_name对应特征
            seq_feat.append(feat)
            seq_label.append(np.asarray(label))


        self.close_h5(crt_feat_list)
        
        seq_feat = np.asarray(seq_feat)
        seq_label = np.asarray(seq_label)
        seq_names = np.asarray(seq_list)

        return seq_feat, seq_label, seq_names



def feat_collate_fn(batch):
    feats, labels, names = list(), list(), list()
    for crt_feat, crt_label, crt_name in batch:
        feats.append(crt_feat)
        labels.append(crt_label)
        names.append(crt_name)

    info = dict()
    info['feat'] = torch.from_numpy(np.asarray(feats)).transpose(0, 1)
    info['label'] = torch.from_numpy(np.asarray(labels)).transpose(0, 1)
    info['name'] = np.asarray(names).T[:,:,None]

    return info
