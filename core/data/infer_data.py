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
class FeatInferDataset(Dataset):
    def __init__(self,
                feat_root,
                img_root,
                seq_len,
                feat_list,
                caption_list = ["upper_cover", "down_cover", "flip", "color"],
                audio_feat_list = ["ecapatdnn", "hubert", "wav2vec", "wav2vec_emotion", "fbank"],
                task='',
                feat_map={}
                ):
        """FeatInferDataset: 根据img_root获取对应的数据集

        Args:
            feat_root (str): 特征h5文件存放目录
            img_root (str): 图像jpg文件存放目录
            seq_len (int): 特征序列长度
            feat_list (list): 需要输入模型的特征文件名 不包括.h5后缀 一维list
            caption_list(opt, list): 特征h5内的帧名存在前缀的list
            audio_feat_list(opt, list): 输入特征中为音频特征的list
            task(opt, str): 数据集任务: train valid test
            feat_map(init, dict): {key:特征名 value:h5特征}
        """

        t0 = time.time()
        self.__dict__.update(locals())
        self.local_rank = torch.distributed.get_rank()
        self.feat_list = [list(_feat.keys())[0] if isinstance(_feat,dict) else _feat for _feat in self.feat_list]
        self.__check_file_exists()

        self.seqfeat_id_list = self.load_seqfeat_id_ds()

        t1 = time.time()
        if self.local_rank == 0:
            logging.info(f" %.1fs for {self.task} dataset initialization." % (t1 - t0))


    def __check_file_exists(self):
        check_file_list =  [self.feat_root, self.img_root]
        for file in check_file_list:
            if not os.path.exists(file):
                logging.info(f'file is not exist. {file}')
        return

    def load_seqfeat_id_ds(self):
        '''获取特征序列id: {video_name/frame_idx}'''

        feat_seq_ids = list()  # 返回序列id list
        for video_name in os.listdir(self.img_root):
            # 读取图像id
            img_ids = list()
            for img_name in os.listdir(os.path.join(self.img_root, video_name)):
                if img_name[-1] == 'g':
                    img_id = int(os.path.splitext(img_name)[0])
                    img_ids.append(img_id)  

            # 图像id排序生成特征id
            img_ids.sort()
            feat_ids = list()
            for img_id in img_ids:
                feat_ids.append(f'{video_name}/{img_id :05}')

            # 特征id序列化
            seq_ids = list()
            seq_len = self.seq_len
            if len(feat_ids) < seq_len:
                feat_ids = (seq_len//len(feat_ids) + 1) * feat_ids
                seq_ids.append(feat_ids[:seq_len])
            else:
                for i in range(0, len(feat_ids), seq_len):
                    seq_ids.append(feat_ids[i: i + seq_len])
                seq_ids[-1] = feat_ids[-seq_len:]

            feat_seq_ids += seq_ids

        return feat_seq_ids


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
                for _cap in self.caption_list:
                    if _cap in feat_name:
                        img_frame_name = f'{_cap}_' + frame_name
                        break
                try:
                    crt_feat = np.asarray(self.feat_map[feat_name][img_frame_name])
                except:
                    video_name = img_frame_name.split('/')[0]
                    frame_id = list(self.feat_map[feat_name][video_name].keys())[-1]
                    crt_feat = np.asarray(self.feat_map[feat_name][f'{video_name}/{frame_id}'])
            feat.append(crt_feat)
        
        return feat


    def __len__(self):
        return len(self.seqfeat_id_list)


    def __getitem__(self, idx):

        crt_feat_list = self.feat_list
        seq_list = self.seqfeat_id_list[idx]
        
        assert len(seq_list) == self.seq_len

        self.open_h5(crt_feat_list)

        seq_feat = list()
        for seq_name in seq_list:
            feat = self.get_frame_feats(seq_name, crt_feat_list) # 获取seq_name对应特征
            feat = np.concatenate(feat, axis=-1)
            seq_feat.append(feat)

        seq_feat = np.asarray(seq_feat)
        seq_names = np.asarray(seq_list)

        self.close_h5(crt_feat_list)

        return seq_feat, seq_names


def infer_feat_collate_fn(batch):
    feats, names = list(), list()
    for crt_feat, crt_name in batch:
        feats.append(crt_feat)
        names.append(crt_name)

    feats = torch.from_numpy(np.asarray(feats)).transpose(0, 1)
    names = np.asarray(names).T[:,:,None]

    return feats, names
