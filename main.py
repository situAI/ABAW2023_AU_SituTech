import random
import torch
import numpy as np
import os

from core.solver import build_solver
from core.utils import args, setup_log

best_prec1 = 0


def init_seed(seed=778):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():

    init_seed(args.seed)
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    out_root = os.path.join(args.out_dir, args.name)
    setup_log(out_root)

    solver = build_solver(args.modality)
    
    solver.run()

    return

if __name__ == '__main__':
    main()
