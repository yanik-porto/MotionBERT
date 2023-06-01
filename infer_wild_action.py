import os
import numpy as np
import argparse
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset
from lib.utils.vismo import render_and_save
from lib.model.model_action import ActionNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/action/MB_ft_NTU60_xsub.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoint/action/MB_ft_NTU60_xsub/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-j', '--json_path', type=str, help='alphapose detection result json path')
    parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    opts = parser.parse_args()
    return opts

opts = parse_args()
args = get_config(opts.config)

model_backbone = load_backbone(args)

model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
if torch.cuda.is_available():
    model = nn.DataParallel(model)
    model = model.cuda()

print('Loading checkpoint', opts.evaluate)
checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()

testloader_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True,
        'drop_last': False
}

vid = imageio.get_reader(opts.vid_path,  'ffmpeg')
fps_in = vid.get_meta_data()['fps']
vid_size = vid.get_meta_data()['size']
os.makedirs(opts.out_path, exist_ok=True)

wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)

test_loader = DataLoader(wild_dataset, **testloader_params)

with torch.no_grad():
    for batch_input in tqdm(test_loader):
        output = model(batch_input)