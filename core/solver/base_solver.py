import os
import tqdm
import numpy as np
import torch
import logging

from core.models import build_model
from core.utils import args, SOLVER_REGISTRY, load_yaml, save_yaml


@SOLVER_REGISTRY.register()
class BaseSolver(object):

    def __init__(self, args=args):

        # init params
        self.epochs = args.epochs
        self.feat_root = args.feat_root
        self.feat_name = args.feat_name
        self.cfg = load_yaml(args.config)
        self.local_rank = torch.distributed.get_rank()
        self.out_root = os.path.join(args.out_dir, args.name)
        self.max_zero_growth_epoch = args.max_zero_growth_epoch
        self.f1_base = args.f1_base
        self.f1_base_list = args.f1_base_list
        self.epoch_base = args.epoch_base
        self.rdrop = args.rdrop
        self.au_list = [1, 2, 4, 6, 7, 10, 12, 15, 23, 24, 25, 26]

        self.best_epoch = 0
        self.val_peek_list = []
        self.best_f1_list = []
        self.best_f1_epochs = []

        # load model
        self.cfg['model']['args']['input_dim'] = sum(list(self.cfg['feat'].values()))
        self.model = build_model(self.cfg['model'])
       
        if 'resume_path' in self.cfg.keys():
            self.load_state(self.cfg['resume_path'])   
            
        if args.local_rank == 0:
            save_yaml(self.cfg, os.path.join(self.out_root, f'config.yaml'))
            logging.info(self.cfg)
        
    def load_state(self, model_path):
        try:
           self.model.load_state_dict(torch.load(model_path))
        except:
            self.model= torch.load(model_path)

        return

    def train(self):
        return

    @torch.no_grad()
    def val(self, e):
        return


    def run(self):
        self.train()
        return

    def save_checkpoint(self, model, save_path, epoch_id, task_name=''):
        model.eval()
        if not task_name:
            torch.save(model, os.path.join(save_path, f'ckpt_{epoch_id}.pth'))
        else:
            torch.save(model, os.path.join(save_path, f'ckpt_{epoch_id}_{task_name}.pth'))
        return