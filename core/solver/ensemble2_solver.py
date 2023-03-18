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
from core.models import build_model

from .ensemble_solver import EnsembleSolver

@SOLVER_REGISTRY.register()
class AuEnsembleSolver(EnsembleSolver):

    def __init__(self, args=args):
        super(AuEnsembleSolver, self).__init__(args)
        self.__check_model()

    def __check_model(self):
        for au_idx in self.csv_title[1:]:
            if au_idx not in self.model_list.keys():
                print(f'lack of {au_idx}')
                assert 1 == 2
                model_dict = self.model_list[au_idx]
                for model_dir in model_dict:
                    model_path = os.path.join(self.model_root, model_dir, model_dict[model_dir])
                    if not os.path.exists(model_path):
                        print(f'no exists {model_path}')
                        assert 1 == 2
        return 
    
    
    def ensemble(self):

        out_preds = list()
        for idx, au_idx in tqdm(enumerate(self.csv_title[1:])):
            logging.info(f'start {au_idx}...')
            model_dict = self.model_list[au_idx]
            au_preds = []
            for model_dir in model_dict:
                crt_feat_dict = self.get_model_feat(model_dir)
                model_path = os.path.join(self.model_root, model_dir, model_dict[model_dir])
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
                    model_id_str = model_dir + '/' + model_dict[model_dir]
                    logging.info(f'{model_id_str}: {_metric_dict}')
                au_preds.append(preds)

            if not idx:
                out_preds = np.asarray(au_preds[0])

            all_preds = np.asarray(au_preds)
            
            if self.entype == 'avg':
                out_preds[:, idx] = all_preds[:, :, idx].mean(axis=0)
            elif self.entype == 'vote':
                out_preds[:, idx] = ((all_preds[:, :, idx] >= self.threshold) * 1).mean(axis=0)
        
        if self.std_out:
            for i in range(out_preds.shape[-1]):
                out_preds[:, i] = (out_preds[:, i] >= self.threshold) * 1
        

        now_time = datetime.datetime.now().strftime("%d-%H-%M-%S")

        if self.ensemble_mode == 'valid':
            metric_dict = self.metric_fn(**{'pred':out_preds, 'gt': labels, 'sigmoid': False})
            logging.info(f'ensemble type: {self.entype}. {metric_dict}')
            f1 = metric_dict['F1']
            pred_name = f'pred_{self.prefix}_{now_time}_' + self.entype + '_' + f'au_ensemble_{f1:.7f}' + '.csv'
            
            gt_name = f'gt_{self.prefix}_{now_time}_' + self.entype + '_' + f'au_ensemble_{f1:.7f}' + '.csv'
            gt_path = os.path.join(self.out_root, gt_name)
            gt_data = dict()
            gt_data[self.csv_title[0]] = names
            for _i, _a in enumerate(self.csv_title[1:]):
                gt_data[_a] = labels[:,_i]
            gt_data = pd.DataFrame.from_dict(gt_data)
            gt_data.to_csv(gt_path, index=False)
        else:
            pred_name = f'{self.prefix}_{now_time}_' + self.entype + '_' + f'au_ensemble' + '.csv'

        pred_data = dict()
        pred_data[self.csv_title[0]] = names
        for _i, _a in enumerate(self.csv_title[1:]):
            pred_data[_a] = out_preds[:,_i]
        pd_data = pd.DataFrame.from_dict(pred_data)
        pred_path = os.path.join(self.out_root, pred_name)
        pd_data.to_csv(pred_path, index=False)

        return
    
