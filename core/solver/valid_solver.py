import os
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
import logging
import time
import pandas as pd
import copy
import glob

from core.data import create_dataloader
from core.utils import args, SOLVER_REGISTRY, load_yaml, save_yaml, format_print_dict
from core.models import build_model, build_metric


@SOLVER_REGISTRY.register()
class ValidSolver(object):

    def __init__(self, args=args):
        super(ValidSolver, self).__init__()

        # init params
        t0 = time.time()
        self.local_rank = torch.distributed.get_rank()
        self.cfg = load_yaml(args.config)
        self.model_root = args.model_root
        self.out_root = os.path.join(args.out_dir, args.name)
        self.model_list = self.cfg['models'] if 'models' in self.cfg.keys() else self.__get_model_list()
        self.report_csv = self.cfg['out_csv'] if 'out_csv' in self.cfg.keys() else 'out.csv'
        self.feat_list = self.cfg['feats']
        self.last_feat_list = []
        for feat in self.cfg['feats'].keys():
            if '_' not in feat:
                self.last_feat_list.append(feat)
        self.csv_title = ['features', 'model_name', 'F1', 'ACC', 'F1_list'] 

        # load metric
        self.metric_fn = build_metric(self.cfg['metric'])
        t1 = time.time()
        if args.local_rank == 0:
            logging.info(' %.1fs for feat solver initialization.'% (t1 - t0))
    

    def __write_csv_file(self, string):

        if not os.path.exists(self.report_csv):
            csv_file = open(self.report_csv, 'a', buffering=1)
            csv_title  = ','.join(self.csv_title) + '\n'
            csv_file.write(csv_title)
        else:
            csv_file = open(self.report_csv, 'a', buffering=1)
    
        csv_file.write(string)
        csv_file.close()

        return


    def __check_in_report(self, string):

        if not os.path.exists(self.report_csv):
            csv_file = open(self.report_csv, 'a', buffering=1)
            csv_title  = ','.join(self.csv_title) + '\n'
            csv_file.write(csv_title)
            return False

        fold_feat_name_list = list()
        csv_file = open(self.report_csv, 'r')
        csv_lines = csv_file.readlines()
        for line in csv_lines:
            fold_feat_name = line.strip('\n').split(',')[0] + ',' + line.strip('\n').split(',')[1]
            fold_feat_name_list.append(fold_feat_name)
            if string in fold_feat_name_list:
                return True

        return False

    def __del_in_report(self, string):

        fold_feat_name_list = list()
        csv_lines = open(self.report_csv, 'r').readlines()
        lines = copy.deepcopy(list(csv_lines))
        del_idx = -1
        for _idx, line in enumerate(csv_lines):
            fold_feat_name = line.strip('\n').split(',')[0] + ',' + line.strip('\n').split(',')[1]
            fold_feat_name_list.append(fold_feat_name)
            if string in fold_feat_name_list:
                del_idx = _idx
                break
        if not del_idx == -1:
            csv_file = open(self.report_csv, 'w', buffering=1)
            for _idx, line in enumerate(lines):
                if _idx == del_idx:
                    continue
                csv_file.write(line)
        return

        
    def __get_model_list(self):

        model_dict = dict()
        model_files = os.path.join(self.model_root, '*')
        for model_dir in glob.glob(model_files):
            model_dir_name = model_dir.split('/')[-1]
            model_list = list()
            for model_name in os.listdir(model_dir):
                if '.pth' in model_name:
                    model_list.append(model_name)
            if model_list:
                model_dict[model_dir_name] = model_list
        
        return model_dict


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
    

    def model_valid(self, cfg):
        try:
            self.model = torch.load(cfg['model_path'])
            self.model.eval()
        except:
            logging.info('load model params...')
            self.model = build_model(cfg['model'])
            try:
                self.model.load_state_dict(torch.load(cfg['model_path']))
            except:
                cfg['model']['args']['seq_len'] = 150
                self.model = build_model(cfg['model'])
                self.model.load_state_dict(torch.load(cfg['model_path']))
            torch.save(self.model, cfg['model_path'])

        self.val_loader = create_dataloader(cfg=cfg)
        
        return self.valid()

    def val(self):

        for model_dir in self.model_list.keys():
            crt_feat_dict = self.get_model_feat(model_dir)
            model_names = [self.model_list[model_dir]] if isinstance(self.model_list[model_dir], str) else self.model_list[model_dir]
            for model_name in model_names:
                model_path = os.path.join(self.model_root, model_dir, model_name)
                if not os.path.exists(model_path):
                    print(f'not exists: {model_path}')
                    continue
                feat_name_with_sep = f'{model_dir},{model_name}'
                if self.__check_in_report(feat_name_with_sep):
                    print(f'\n exists {feat_name_with_sep}')
                    continue
                else:
                    self.__write_csv_file(string=f'{feat_name_with_sep}\n')
                

                logging.info(f'Start {model_dir}/{model_name} {crt_feat_dict} ...')
                crt_cfg = copy.deepcopy(self.cfg)
                crt_cfg['model_path'] = model_path 
                crt_cfg['feat'] = crt_feat_dict
                crt_cfg['model']['args']['input_dim'] = sum(list(crt_feat_dict.values()))
                f1, acc, f1_list = self.model_valid(crt_cfg)
                self.__del_in_report(feat_name_with_sep)
                self.__write_csv_file(f'{model_dir},{model_name},{f1},{acc},{f1_list}\n')
                logging.info(f'End {model_dir}/{model_name} {crt_feat_dict} ...')
        return
    
    
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

        metric_dict = self.metric_fn(**{'pred': np.asarray(df['pred'].tolist()), 'gt': np.asarray(df['label'].tolist())})

        print_dict = dict()
        print_dict.update(metric_dict)

        if torch.distributed.get_rank() == 0:
            logging.info(f"==> val: {print_dict}")
        
        return metric_dict['F1'], metric_dict['ACC'],  metric_dict['F1_list']
        
    
    def run(self):
        self.val()
        return
