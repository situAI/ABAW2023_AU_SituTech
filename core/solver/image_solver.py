import os
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from core.data import create_dataloader
from core.utils import args, SOLVER_REGISTRY, load_yaml, save_yaml
from core.models import build_loss, build_metric, build_optimizer, build_lr_scheduler
from .base_solver import BaseSolver
import logging
import time


@SOLVER_REGISTRY.register()
class ImageSolver(BaseSolver):

    def __init__(self, args=args):
        super(ImageSolver, self).__init__(args)

        # load data
        t0 = time.time()
        self.train_loader, self.val_loader = create_dataloader()
        self.len_train_loader, self.len_val_loader = len(self.train_loader), len(self.val_loader)
        self.evaluate_period =  self.len_train_loader // 16
        self.evaluate_sample =  self.len_val_loader // 8
        
        # load model
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DistributedDataParallel(self.model.cuda(self.local_rank), device_ids=[self.local_rank], find_unused_parameters=True)
        
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
        
        val_peek_list = [-1]
        for e in range(self.epochs):
            if torch.distributed.get_rank() == 0:
                logging.info(f'==> Start Training epoch {e}')
            
            step = 0
            pred_list = list()
            label_list = list()
            mean_loss = 0.0
            for imgs, labels in iter(tqdm(self.train_loader)):
                self.optimizer.zero_grad()

                imgs, labels =  imgs.cuda(self.local_rank), labels.cuda(self.local_rank)
    
                preds = self.model(imgs)
                seq_len, bs= preds.shape

                loss = 0
                for _loss in self.loss_list:
                    loss += _loss(preds, labels.float())
                mean_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()

                if (step == 0 or step % 100 == 0) and (torch.distributed.get_rank() == 0):
                    logging.info(f'epoch: {e + 1}/{self.epochs}, iteration: {step + 1}/{self.len_train_loader}, loss: {loss.item() :.4f}')
                
                if (step != 0 and step % self.evaluate_period == 0) and (torch.distributed.get_rank() == 0):
                    crt_e = e+1 + round(step / self.len_train_loader, 1)
                    peek =self.val(crt_e)
                    if peek > max(val_peek_list):
                        if peek > self.f1_base:
                            peek_str = str(round(peek * 100, 2)).replace('.', '_')
                            self.save_checkpoint(self.model, self.out_root, f'{e+1}_{step}_f1_{peek_str}')
                        val_peek_list.append(peek)
                        best_epoch = crt_e
                    self.model.train()
                
                batch_pred = [torch.zeros_like(preds) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(batch_pred, preds)
                pred_list.append(torch.cat(batch_pred, dim=0).detach().cpu())

                batch_label = [torch.zeros_like(labels) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(batch_label, labels)
                label_list.append(torch.cat(batch_label, dim=0).detach().cpu())

                step += 1
                
            pred_list = torch.cat(pred_list, dim=0)
            label_list = torch.cat(label_list, dim=0)
            pred_list = pred_list.numpy()
            label_list = label_list.numpy()
            metric_dict = self.metric_fn(**{'pred': pred_list, 'gt': label_list})
            step_mean_loss = mean_loss / step
            
            print_dict = dict()
            print_dict.update({'epoch': f'{e + 1}/{self.epochs}'})
            print_dict.update({"iteration": f'{step + 1}/{self.len_train_loader}'})
            print_dict.update({'mean_loss': step_mean_loss})
            print_dict.update({'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})
            print_dict.update(metric_dict)
                
            if torch.distributed.get_rank() == 0:
                logging.info(f"==> train: {print_dict}")
                peek = self.val(e + 1)
                if peek > max(val_peek_list):
                    self.save_checkpoint(self.model, self.out_root, f'{e+1}_f1_{peek_str}')
                    val_peek_list.append(peek)
                self.model.train()
            
            if self.cfg['lr_scheduler']['name'] in ["CosineLRScheduler"]:
                lr_scheduler.step(e)
            else:
                lr_scheduler.step()

        if self.local_rank == 0:
            logging.info(f'==> End Training, BEST F1: {max(val_peek_list)}')
            return max(val_peek_list)
    

    @torch.no_grad()
    def val(self, e):
        self.model.eval()

        all_preds = list()
        all_labels = list()

        val_num = 0
        for imgs, labels in iter(self.val_loader):
            if val_num in range(self.evaluate_sample):
                val_num += 1
            else:
                continue
            labels = labels.cpu().detach().numpy()
            preds = self.model(imgs).cpu().detach().numpy()
            all_preds = np.concatenate([all_preds, preds],axis=0) if len(all_preds) else preds
            all_labels = np.concatenate([all_labels, labels],axis=0) if len(all_labels) else labels

        metric_dict = self.metric_fn(**{'pred': all_preds, 'gt': all_labels})
        print_dict = dict()
        print_dict.update({'epoch': e})
        print_dict.update(metric_dict)

        crt_f1 = metric_dict['F1_list']
        if len(self.best_f1_list) == 0:
            self.best_f1_list = crt_f1
            self.best_f1_epochs = [1 for _ in range(len(crt_f1))]
        else:
            for _i in range(len(crt_f1)):
                if self.best_f1_list[_i] < crt_f1[_i]:
                    self.best_f1_list[_i] = crt_f1[_i]
                    self.best_f1_epochs[_i] = e

        if torch.distributed.get_rank() == 0:
           logging.info(f"==> val: {print_dict}")
           logging.info(f"====> val best f1 avg: {sum(self.best_f1_list)/len(self.best_f1_list)} f1_list: {self.best_f1_list}, f1_epoch_list: {self.best_f1_epochs}")

        peek = metric_dict['F1']

        return peek