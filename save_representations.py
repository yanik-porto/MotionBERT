import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from lib.data.dataset_action import NTURGBD
from lib.utils.learning import *
from lib.utils.tools import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    opts = parser.parse_args()
    return opts

def save_rep(args, opts):
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    data_path = 'data/action/%s.pkl' % args.dataset
    ntu60_xsub_train = NTURGBD(data_path=data_path, data_split=args.data_split+'_train', n_frames=args.clip_len, random_move=args.random_move, scale_range=args.scale_range_train)
    ntu60_xsub_val = NTURGBD(data_path=data_path, data_split=args.data_split+'_val', n_frames=args.clip_len, random_move=False, scale_range=args.scale_range_test)
    train_loader = DataLoader(ntu60_xsub_train, **trainloader_params)
    test_loader = DataLoader(ntu60_xsub_val, **testloader_params)

    model_backbone = load_backbone(args)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    model_backbone.eval()
    for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):
        batch_size = len(batch_input)        
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()

        N, M, T, J, C = batch_input.shape
        batch_input = batch_input.reshape(N*M, T, J, C)
        batch_rep = model_backbone(batch_input, return_rep=True)
        # batch_rep = model_backbone.get_representation(batch_input)
        print(batch_rep.shape)

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    save_rep(args, opts)
