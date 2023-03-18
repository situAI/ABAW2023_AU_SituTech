import os
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
import logging
import time
import pandas as pd
import copy

from core.data import create_dataloader
from core.utils import args, SOLVER_REGISTRY, load_yaml, save_yaml, format_print_dict
from core.models import build_model


@SOLVER_REGISTRY.register()
class InferSolver(object):

    def __init__(self, args=args):
        super(InferSolver, self).__init__()

        # init params
        t0 = time.time()
        self.local_rank = torch.distributed.get_rank()
        self.cfg = load_yaml(args.config)
        self.model_root = args.model_root
        self.out_root = os.path.join(args.out_dir, args.name)
        self.model_list = self.cfg['models']
        self.feat_list = self.cfg['feats']
        self.last_feat_list = []
        for feat in self.cfg['feats'].keys():
            if '_' not in feat:
                self.last_feat_list.append(feat)
        self.csv_title = ['frame_id', 'AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26'] 
        t1 = time.time()
        if args.local_rank == 0:
            logging.info(' %.1fs for feat solver initialization.'% (t1 - t0))
    
    
    def get_model_feat(self, model_dir):
        '''获取模型对应需要的feat'''
        model_feat = model_dir
        try:
            fold = int(model_feat[0])
            model_feat = model_feat[1:]
        except:
            model_feat = model_feat
        model_feat = model_feat.replace(';', '_')
        for fidx, feat in enumerate(self.feat_list.keys()):
            if feat in model_feat and feat not in self.last_feat_list:
                model_feat = model_feat.replace(feat, f'{fidx}')
        
        for fidx, feat in enumerate(self.feat_list.keys()):
            if feat in self.last_feat_list:
                model_feat = model_feat.replace(feat, f'{fidx}')
        
        feat_dict = dict()
        feat_list = list(self.feat_list.keys())
        for model_idx in model_feat.split('_'):
            try:
                model_idx = int(model_idx)
            except:
                print(f'warning: model name: {model_dir} model feat:{model_feat} error model idx: {model_idx}')
                continue
            crt_feat = feat_list[model_idx]
            feat_dict[crt_feat] = self.feat_list[crt_feat]
        
        return feat_dict
    

    def model_infer(self, cfg):

        self.model = torch.load(cfg['model_path'])

        self.val_loader = create_dataloader(cfg=cfg)
        
        return self.test()
    
    
    def infer(self):

        for model_dir in tqdm(self.model_list.keys()):
            crt_feat_dict = self.get_model_feat(model_dir)
            model_path = os.path.join(self.model_root, model_dir, self.model_list[model_dir])
            if not os.path.exists(model_path):
                print(f'not exists: {model_path}')
                continue

            crt_cfg = copy.deepcopy(self.cfg)
            crt_cfg['model_path'] = model_path 
            crt_cfg['feat'] = crt_feat_dict
            all_preds, all_names = self.model_infer(crt_cfg)

            out_data = dict()
            out_data[self.csv_title[0]] = all_names
            for _i, _a in enumerate(self.csv_title[1:]):
                out_data[_a] = all_preds[:,_i]
            pd_data = pd.DataFrame.from_dict(out_data)
            pd_data = pd_data.drop_duplicates(subset=self.csv_title[0], keep='first')

            csv_name = model_dir + '_' + os.path.splitext(self.model_list[model_dir])[0] + '.csv'
            csv_path = os.path.join(self.out_root, csv_name)
            pd_data.to_csv(csv_path, index=False)
            print(f'infer done. {csv_path}')

        return

        
    @torch.no_grad()
    def test(self):
        self.model.eval()

        all_preds, all_names = list(), list()
        for imgs, names in tqdm(iter(self.val_loader)):
            preds = self.model(imgs).detach()
            preds = torch.sigmoid(preds).cpu()
            seq_len, bs, _= preds.shape
            preds = preds.reshape((seq_len * bs, -1)).numpy()
            names = names.reshape((seq_len * bs, -1))
            
            all_preds = np.concatenate([all_preds, preds],axis=0) if len(all_preds) else preds
            all_names = np.concatenate([all_names, names],axis=0) if len(all_names) else names
    
        return all_preds, all_names.squeeze()
    
    def run(self):
        self.infer()
        return
