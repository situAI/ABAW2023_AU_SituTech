import os
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
import logging
import time
import pandas as pd

from core.data import create_dataloader
from core.utils import args, SOLVER_REGISTRY, load_yaml, save_yaml, format_print_dict
from core.models import build_model, build_loss, build_metric, build_optimizer, build_lr_scheduler
from .base_solver import BaseSolver


@SOLVER_REGISTRY.register()
class FeatSolver(BaseSolver):

    def __init__(self, args=args):
        super(FeatSolver, self).__init__(args)

        t0 = time.time()
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DistributedDataParallel(self.model.cuda(self.local_rank), device_ids=[self.local_rank], find_unused_parameters=True)

        # load data
        self.train_loader, self.val_loader = create_dataloader(cfg=self.cfg)
        self.len_train_loader, self.len_val_loader = len(self.train_loader), len(self.val_loader)
        self.show_loss_period =  max(self.len_train_loader // 3, 1)
        self.evaluate_period =  max(self.len_train_loader // 2, 1)

        # load optimizer
        self.optimizer = build_optimizer(self.model.parameters(), self.cfg['optimizer'])
        
        # load loss
        self.loss_list = []
        for key in self.cfg.keys():
            if 'loss' in key:
                self.loss_list.append(build_loss(self.cfg[key]).cuda(self.local_rank))

        # load metric
        self.metric_fn = build_metric(self.cfg['metric'])

        t1 = time.time()
        if args.local_rank == 0:
            logging.info(' %.1fs for feat solver initialization.'% (t1 - t0))

    
    def train(self):
        
        lr_scheduler = build_lr_scheduler(self.optimizer, self.cfg['lr_scheduler'])
        
        for e in range(self.epochs):
            self.train_loader.sampler.set_epoch(e)
            if torch.distributed.get_rank() == 0:
                logging.info(f'==> Start Training epoch {e+1}')

            self.model.train()
            
            step = 0
            pred_list = list()
            label_list = list()
            mean_loss = 0.0
            for info in iter(tqdm(self.train_loader)):
                feat = info['feat']
                labels = info['label']

                self.optimizer.zero_grad()
                feat, labels =  feat.cuda(self.local_rank), labels.cuda(self.local_rank)

                preds = self.model(feat)
                # print(preds.shape, labels.shape)

                seq_len, bs, _ = preds.shape
                preds = preds.reshape((seq_len * bs, -1))
                labels = labels.reshape((seq_len * bs, -1))

                if self.rdrop:
                    preds2 = self.model(feat)
                    preds2 = preds2.reshape((seq_len * bs, -1))
                    loss = 0
                    for _loss in self.loss_list:
                        loss += _loss(preds, preds2, labels.float())
                    mean_loss += loss.item()
                else:
                    loss = 0
                    for _loss in self.loss_list:
                        loss += _loss(preds, labels.float())
                    mean_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                
                if (step == 0 or step % self.show_loss_period == 0) and (torch.distributed.get_rank() == 0):
                    crt_e = e + round(step / self.len_train_loader, 1)
                    logging.info(f'epoch: {crt_e}/{self.epochs}, iteration: {step + 1}/{self.len_train_loader}, loss: {loss.item() :.4f}')
                
                if (step != 0 and step % self.evaluate_period == 0 and step != self.len_train_loader - 1) and (torch.distributed.get_rank() == 0):
                    crt_e = e + round(step / self.len_train_loader, 1)
                    self.val(crt_e)
                    self.model.train()

                batch_pred = [torch.zeros_like(preds) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(batch_pred, preds)
                pred_list.append(torch.cat(batch_pred, dim=0).detach().cpu())

                batch_label = [torch.zeros_like(labels) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(batch_label, labels)
                label_list.append(torch.cat(batch_label, dim=0).detach().cpu())

                step += 1
            
            all_preds = torch.cat(pred_list, dim=0).numpy()
            all_labels = torch.cat(label_list, dim=0).numpy()

            metric_dict = self.metric_fn(**{'pred':all_preds, 'gt': all_labels})
            mean_loss = mean_loss / self.len_train_loader

            print_dict = dict()
            print_dict.update({'epoch': f'{e + 1}/{self.epochs}'})
            print_dict.update({'mean_loss': mean_loss})
            print_dict.update({'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})
            print_dict.update(metric_dict)
            print_str = format_print_dict(print_dict)

            if torch.distributed.get_rank() == 0:
                logging.info(f"==> train: {print_str}")
                self.val(e + 1)
                max_best_epoch = max(self.best_f1_epochs + [self.best_epoch])
                if int(e+1 - max_best_epoch) > self.max_zero_growth_epoch :
                    logging.info(f'==> End Training, No new best_epoch. crt epoch: {e+1} best_epoch: {self.best_epoch}')
                    break
                elif (e + 1) > self.epoch_base and max(self.val_peek_list) < self.f1_base:
                    logging.info(f'==> End Training, Below F1_base: {self.f1_base}. best f1: {max(self.val_peek_list)}')
                    break

            lr_scheduler.step()

        if self.local_rank == 0:
            logging.info(f'==> End Training, BEST F1: {max(self.val_peek_list)} BEST epoch: {self.best_epoch}')
    
        return max(self.val_peek_list), self.best_epoch, sum(self.best_f1_list)/12, list(self.best_f1_list), list(self.best_f1_epochs)
    

    @torch.no_grad()
    def val(self, e):
        self.model.eval()

        all_preds, all_labels, all_names = list(), list(), list()

        for info in iter(self.val_loader):

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
        print_dict.update({'epoch': e})
        print_dict.update(metric_dict)

        acc = metric_dict['ACC']
        peek = metric_dict['F1']
        crt_f1_list = metric_dict['F1_list']
        # 初始化best f1
        if len(self.best_f1_list) == 0:  
            self.best_epoch = e
            self.val_peek_list.append(peek)
            self.best_f1_list = crt_f1_list
            self.best_f1_epochs = [e for _ in range(len(crt_f1_list))]
        
        # 保存高于f1_base_list F1
        for _i in range(len(crt_f1_list)):
            if crt_f1_list[_i] > self.best_f1_list[_i]:
                if crt_f1_list[_i] > self.f1_base_list[_i]:
                    au_num = self.au_list[_i]
                    au_f1 = str(round(crt_f1_list[_i] * 100, 2)).replace('.', '_')
                    peek_str = str(round(peek * 100, 0)).replace('.', '_')
                    self.save_checkpoint(self.model, self.out_root,  f'{e}_F1_{peek_str}_AU{au_num}_F1_{au_f1}')
                self.best_f1_list[_i] = crt_f1_list[_i]
                self.best_f1_epochs[_i] = e
            
        # 保存高于f1_base F1
        if peek >= max(self.val_peek_list):
            if peek > self.f1_base:
                acc_str = str(round(acc * 100, 1)).replace('.', '_')
                peek_str = str(round(peek * 100, 3)).replace('.', '_')
                self.save_checkpoint(self.model, self.out_root,  f'{e}_F1_{peek_str}_acc_{acc_str}')
            elif int(e) == float(e):
                acc_str = str(round(acc * 100, 1)).replace('.', '_')
                peek_str = str(round(peek * 100, 3)).replace('.', '_')
                self.save_checkpoint(self.model, self.out_root,  f'{e}_F1_{peek_str}_acc_{acc_str}')
            self.val_peek_list.append(peek)
            self.best_epoch = e

        if torch.distributed.get_rank() == 0:
            logging.info(f"==> val: {print_dict}")
            logging.info(f"====> val best F1 avg: {sum(self.best_f1_list)/len(self.best_f1_list)} F1_list: {self.best_f1_list}, F1_epoch_list: {self.best_f1_epochs}")

        return peek
