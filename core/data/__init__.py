from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..utils import args
from .image_data import ImageDataset, ABAW5
from .feat_data import FeatSeqDataset, feat_collate_fn
from .infer_data import FeatInferDataset, infer_feat_collate_fn

def create_dataloader(cfg='', modality=args.modality, image_size=args.image_size, batch_size=args.batch_size, workers=args.workers, seq_len=args.seq_len, label_mode=args.label_mode):

    if modality == 'ImageSolver':

        train_datasets = ImageDataset(image_root=args.image_root, label_root=args.train_label_root, paral_image_roots=args.paral_image_roots,image_size=image_size, task="train")
        val_datasets = ImageDataset(image_root=args.image_root, label_root=args.val_label_root, paral_image_roots=args.paral_image_roots, image_size=image_size, task="valid")
        
        train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers, drop_last=False)
        val_loader = DataLoader(val_datasets, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers, drop_last=False)
        
        return train_loader, val_loader

    elif modality == 'InferSolver' or (modality in ['EnsembleSolver', 'AuEnsembleSolver'] and args.ensemble_mode=='infer'):

        feat_list = cfg['feat'] if isinstance(cfg['feat'], list) else list(cfg['feat'].keys()) if 'feat' in cfg.keys() else []
      
        val_datasets = FeatInferDataset(feat_root=args.feat_root, img_root=args.image_root, seq_len=seq_len, feat_list=feat_list, task="test")
        val_loader = DataLoader(val_datasets, collate_fn=infer_feat_collate_fn, batch_size=batch_size, pin_memory=True, num_workers=workers, drop_last=False)
       
        return val_loader

    elif modality in ['ValidSolver'] or (modality in ['EnsembleSolver', 'AuEnsembleSolver'] and args.ensemble_mode=='valid'):

        feat_list = cfg['feat'] if isinstance(cfg['feat'], list) else list(cfg['feat'].keys()) if 'feat' in cfg.keys() else []
      
        val_datasets = FeatSeqDataset(feat_root=args.feat_root, label_root=args.val_label_root, img_root=args.image_root, seq_len=seq_len, feat_list=feat_list, task="valid")
        val_loader = DataLoader(val_datasets, collate_fn=feat_collate_fn, batch_size=batch_size, pin_memory=True, num_workers=workers, drop_last=False)

        return val_loader

    else:  # feat

        feat_list = cfg['feat'] if isinstance(cfg['feat'], list) else list(cfg['feat'].keys()) if 'feat' in cfg.keys() else []

        train_datasets = FeatSeqDataset(feat_root=args.feat_root, label_root=args.train_label_root, img_root=args.image_root, seq_len=seq_len, feat_list=feat_list, task="train")
        val_datasets = FeatSeqDataset(feat_root=args.feat_root, label_root=args.val_label_root, img_root=args.image_root, seq_len=seq_len, feat_list=feat_list, task="valid")

        train_sampler = DistributedSampler(train_datasets)

        train_loader = DataLoader(train_datasets, sampler=train_sampler, collate_fn=feat_collate_fn, batch_size=batch_size, pin_memory=True, num_workers=workers, drop_last=False)
        val_loader = DataLoader(val_datasets, collate_fn=feat_collate_fn, batch_size=batch_size, pin_memory=True, num_workers=workers, drop_last=False)

        return train_loader, val_loader


def creat_abaw5_loader(root=args.image_root, image_size=args.image_size, batch_size=args.batch_size, workers=args.workers, caption=''):
    
    image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
          
    ds = ABAW5(data_root=root, image_size=image_size, caption=caption)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=workers, drop_last=False)

    return dl