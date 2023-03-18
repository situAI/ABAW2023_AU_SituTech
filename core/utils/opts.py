import os
import argparse

parser = argparse.ArgumentParser(description="AU Detection")

# ========================= Data Configs ==========================

parser.add_argument('config',type=str, help="model config file")
parser.add_argument('modality', type=str, help='Solver type')
parser.add_argument('--image_root', type=str, default="/data1/ABAW/ABAW5/Aff-Wild2/cropped_aligned_images/cropped_aligned/")
parser.add_argument('--image_size', type=int, default=112)
parser.add_argument('--feat_root', type=str, default="/data1/ABAW/ABAW5/Aff-Wild2/feat")
parser.add_argument('--feat_name', type=str, help="save feat name when extract feature")
parser.add_argument('--seq_len', type=int, default=128)
parser.add_argument('--label_root', type=str, default="/data1/ABAW/ABAW5/Aff-Wild2/annotations/AU_Detection_Challenge/All_Set")
parser.add_argument('--train_label_root', type=str, default="/data1/ABAW/ABAW5/Aff-Wild2/annotations/AU_Detection_Challenge/Train_Set")
parser.add_argument('--val_label_root', type=str, default="/data1/ABAW/ABAW5/Aff-Wild2/annotations/AU_Detection_Challenge/Validation_Set")
parser.add_argument('--label_mode', type=str, default="all", choices=['upper_cover', 'down_cover', 'all'])
parser.add_argument('--paral_image_roots', type=str, nargs='*', help='the same label with image_root')

# # ========================= Traning Configs ==========================

parser.add_argument('--kfold_json', type=str, default='/data1/ABAW/ABAW5/Aff-Wild2/kfold/au.json')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run (default: 50)')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--batch-size', default=32, type=int, help='mini-batch size')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--out_dir', type=str, default='./runs', help='path to save outputs')
parser.add_argument('--rdrop', action='store_true', help='double inference for calulate loss' )
parser.add_argument('--name', type=str, default='exp', help='saved to output_dir/name' )
parser.add_argument('--seed', type=int, default=3400)

parser.add_argument('--max_zero_growth_epoch', type=int, default=1)
parser.add_argument('--f1_base', type=float, default=0.54)
# parser.add_argument('--f1_base_list', type=list, default=[0.58, 0.52, 0.65, 0.69, 0.77, 0.76, 0.772, 0.35, 0.22, 0.30, 0.86, 0.43])
parser.add_argument('--f1_base_list', type=list, default=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
parser.add_argument('--epoch_base', type=int, default=6)

# # ========================= Inference Configs ==========================
parser.add_argument('--model_root', type=str, default="/home/dyy/project/ABAW5/au_abaw5/search_runs/0308")
parser.add_argument('--ensemble', type=str, default="avg", choices=['avg', 'vote'])
parser.add_argument('--ensemble_mode', type=str, default="valid", choices=['infer', 'valid'], help='')
parser.add_argument('--threshold', type=float, default=0.5, help='ensemble label threshold')
parser.add_argument('--std_out', action='store_true', help='output: 0 or 1' )
parser.add_argument('--prefix', type=str, default="", help='ensemble csv prefix' )

args = parser.parse_args()
