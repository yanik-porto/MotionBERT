import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from lib.data.dataset_action import NTURGBD
from lib.utils.learning import *
from lib.utils.tools import *
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint file')
    parser.add_argument('-gpulog', '--log_gpu_memory', default=False, action='store_true', help='log gpu memory usage')
    opts = parser.parse_args()
    return opts

def load_model(args, opts):
    model_backbone = load_backbone(args)
    print('Loading backbone', opts.pretrained)
    checkpoint = torch.load(opts.pretrained, map_location=lambda storage, loc: storage)['model_pos']
    model_backbone = load_pretrained_weights(model_backbone, checkpoint)

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    return model_backbone

def save_dataset(dataloader, model_backbone, pickle_file, max_batch):
    print('INFO: Running on {} batches'.format(len(dataloader)))

    datarep = []

    model_backbone.eval()
    for idx, (batch_input, batch_gt) in tqdm(enumerate(dataloader)):
        batch_size = len(batch_input)
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()

        if opts.log_gpu_memory:
            print_gpu_memory()

        N, M, T, J, C = batch_input.shape
        batch_input = batch_input.reshape(N*M, T, J, C)
        batch_rep = model_backbone(batch_input, return_rep=True)
        # batch_rep = model_backbone.get_representation(batch_input)

        batch_rep_cpu = batch_rep.cpu().detach().numpy()
        batch_gt_cpu = batch_gt.cpu().detach().numpy()
        datarep.append({'rep':batch_rep_cpu, 'gt':batch_gt_cpu})
        batch_rep = None

        if idx == max_batch:
            break
    
    with open(pickle_file, 'wb') as f:
        pickle.dump(datarep, f)

def save_train_rep(args, opts, model_backbone):
    print('Loading dataset...')
    if opts.log_gpu_memory:
        print_gpu_memory()

    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 4,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    data_path = 'data/action/%s.pkl' % args.dataset
    ntu60_xsub_train = NTURGBD(data_path=data_path, data_split=args.data_split+'_train', n_frames=args.clip_len, random_move=args.random_move, scale_range=args.scale_range_train)
    train_loader = DataLoader(ntu60_xsub_train, **trainloader_params)

    print("train_loader contains ", len(train_loader))

    print('...loaded')
    if opts.log_gpu_memory:
        print_gpu_memory()

    save_dataset(train_loader, model_backbone, 'train_data.pickle', 100)


def save_test_rep(args, opts, model_backbone):
    print('Loading dataset...')
    if opts.log_gpu_memory:
        print_gpu_memory()

    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 4,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    data_path = 'data/action/%s.pkl' % args.dataset
    ntu60_xsub_val = NTURGBD(data_path=data_path, data_split=args.data_split+'_val', n_frames=args.clip_len, random_move=False, scale_range=args.scale_range_test)
    test_loader = DataLoader(ntu60_xsub_val, **testloader_params)

    print("test_loader contains ", len(test_loader))

    print('...loaded')
    if opts.log_gpu_memory:
        print_gpu_memory()

    save_dataset(test_loader, model_backbone, 'test_data.pickle', 10)

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    args.epochs = 1
    args.batch_size = 1

    if opts.log_gpu_memory:
        print_gpu_memory()
    model_backbone = load_model(args, opts)
    if opts.log_gpu_memory:
        print_gpu_memory()

    save_test_rep(args, opts, model_backbone)
    save_train_rep(args, opts, model_backbone)
