import os
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
import logging
import time
import pandas as pd
import datetime
import copy

from core.data import create_dataloader
from core.utils import args, SOLVER_REGISTRY, load_yaml, save_yaml, format_print_dict
from core.models import build_model, build_metric

from .infer_solver import InferSolver


@SOLVER_REGISTRY.register()
class EnsembleSolver(InferSolver):

    def __init__(self, args=args):
        super(EnsembleSolver, self).__init__(args)
        self.entype = args.ensemble
        self.threshold = args.threshold
        self.ensemble_mode = args.ensemble_mode
        self.std_out = args.std_out
        self.prefix = args.prefix
        if self.ensemble_mode == 'valid':
            self.metric_fn = build_metric(self.cfg['metric'])
    
    def ensemble(self):

        all_preds = list()
        for model_dir in tqdm(self.model_list.keys()):
            crt_feat_dict = self.get_model_feat(model_dir)
            model_path = os.path.join(self.model_root, model_dir, self.model_list[model_dir])
            model_id_str = model_dir + '/' + self.model_list[model_dir]
            if not os.path.exists(model_path):
                print(f'not exists: {model_path}')
                continue

            crt_cfg = copy.deepcopy(self.cfg)
            crt_cfg['model_path'] = model_path 
            crt_cfg['feat'] = crt_feat_dict
            crt_cfg['model']['args']['input_dim'] = sum(list(crt_feat_dict.values()))
            if self.ensemble_mode == 'infer':
                preds, names = self.model_infer(crt_cfg)
            elif self.ensemble_mode == 'valid':
                preds, labels, names = self.model_valid(crt_cfg)
                _metric_dict = self.metric_fn(**{'pred':preds, 'gt': labels, 'sigmoid': False})
                logging.info(f'{model_id_str}: {_metric_dict}')
           
            all_preds.append(preds)
            logging.info(f'{model_id_str} done.')

        all_preds = np.asarray(all_preds)

        out_preds = np.zeros(all_preds[0].shape)
        if self.entype == 'avg':
            for i in range(all_preds.shape[-1]):
                out_preds[:, i] = all_preds[:, :, i].mean(axis=0)
        elif self.entype == 'vote':
            for i in range(all_preds.shape[-1]):
                out_preds[:, i] = ((all_preds[:, :, i] >= self.threshold) * 1).mean(axis=0)
        
        if self.std_out:
            for i in range(preds.shape[-1]):
                out_preds[:, i] = (out_preds[:, i] >= self.threshold) * 1
        
        now_time = datetime.datetime.now().strftime("%d-%H-%M-%S")

        if self.ensemble_mode == 'valid':
            metric_dict = self.metric_fn(**{'pred':out_preds, 'gt': labels, 'sigmoid': False})
            logging.info(f'ensemble type: {self.entype}. {metric_dict}')
            f1 = metric_dict['F1']
            pred_name = f'pred_{self.prefix}_{now_time}_' + self.entype + '_' + f'ensemble_{f1:.7f}' + '.csv'
            
            gt_name = f'gt_{self.prefix}_{now_time}_' + self.entype + '_' + f'ensemble_{f1:.7f}' + '.csv'
            gt_data = dict()
            gt_data[self.csv_title[0]] = names
            for _i, _a in enumerate(self.csv_title[1:]):
                gt_data[_a] = labels[:,_i]
            gt_data = pd.DataFrame.from_dict(gt_data)
            gt_path = os.path.join(self.out_root, gt_name)
            gt_data.to_csv(gt_path, index=False)
        else:
            pred_name = f'{self.prefix}_{now_time}_' + self.entype + '_' + f'ensemble' + '.csv'

        pred_data = dict()
        pred_data[self.csv_title[0]] = names
        for _i, _a in enumerate(self.csv_title[1:]):
            pred_data[_a] = out_preds[:,_i]
        pd_data = pd.DataFrame.from_dict(pred_data)
        pred_path = os.path.join(self.out_root, pred_name)
        pd_data.to_csv(pred_path, index=False)

        return
    

    def model_valid(self, cfg):
        try:
            self.model = torch.load(cfg['model_path'])
            self.model.eval()
        except:
            logging.info('load model params...')
            self.model = build_model(cfg['model'])
            self.model.load_state_dict(torch.load(cfg['model_path']))
            torch.save(self.model, cfg['model_path'])

        self.val_loader = create_dataloader(cfg=cfg)
        
        return self.valid()


    @torch.no_grad()
    def valid(self):
        self.model.eval()

        all_preds, all_labels, all_names = list(), list(), list()

        for info in tqdm(iter(self.val_loader)):

            feat = info['feat']
            labels = info['label']
            names = info['name']
        
            labels = labels.detach().cpu()
            preds = self.model(feat).detach().cpu()

            seq_len, bs, _= preds.shape

            preds = preds.reshape((seq_len * bs, -1)).numpy()
            labels = labels.reshape((seq_len * bs, -1)).numpy()
            names = names.reshape((seq_len * bs, -1))

            all_preds = np.concatenate([all_preds, preds],axis=0) if len(all_preds) else preds
            all_labels = np.concatenate([all_labels, labels],axis=0) if len(all_labels) else labels
            all_names = np.concatenate([all_names, names],axis=0) if len(all_names) else names
    
        df = dict()
        df['seq_name'] = all_names.squeeze()
        df['pred'] = all_preds
        df['label'] = all_labels
        df = pd.DataFrame.from_dict(df, orient='index').T
        df = df.drop_duplicates(subset=['seq_name'], keep='first')
        preds, labels, names = np.asarray(df['pred'].tolist()), np.asarray(df['label'].tolist()), np.asarray(df['seq_name'].tolist())
        
        preds = torch.tensor(preds)
        preds = torch.sigmoid(preds)
        preds = np.array(preds)

        return preds, labels, names
        
    
    def run(self):
        self.ensemble()
        return
