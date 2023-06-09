import argparse
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from lib.model.model_action import ActionHeadEmbed
from lib.data.dataset_motion_rep import MotionDatasetRep
from lib.utils.tools import *
from lib.utils.learning import *
import errno

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=100)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-gpulog', '--log_gpu_memory', default=False, action='store_true', help='log gpu memory usage')
    opts = parser.parse_args()
    return opts

def train_with_config(args, opts):
    print(args)
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)

    print("Load model")
    model = ActionHeadEmbed(dropout_ratio=args.dropout_ratio, dim_rep=args.dim_rep, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda() 
    if opts.log_gpu_memory:
        print_gpu_memory()

    best_acc = 0
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    print('Loading dataset...')
    if opts.log_gpu_memory:
        print_gpu_memory()
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }

    ntu60_xsub_train = MotionDatasetRep(file_path=args.dataset)
    train_loader = DataLoader(ntu60_xsub_train, **trainloader_params)
    print('...loaded')
    if opts.log_gpu_memory:
        print_gpu_memory()

    optimizer = optim.AdamW(
        [{"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": args.lr_head},
        ],      lr=args.lr_head, 
                weight_decay=args.weight_decay
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    for epoch in range(args.epochs):
        print('Training epoch %d.' % epoch)
        losses_train = AverageMeter()
        model.train()

        iters = len(train_loader)
        for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):    # (N, 2, T, 17, 3)
            batch_size = len(batch_input)
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            if opts.log_gpu_memory:
                print("after batch load")
                print_gpu_memory()

            output = model(batch_input) # (N, num_classes)
            if opts.log_gpu_memory:
                print("after inference")
                print_gpu_memory()

            optimizer.zero_grad()
            loss_train = criterion(output, batch_gt)
            losses_train.update(loss_train.item(), batch_size)
            loss_train.backward()
            optimizer.step()

            scheduler.step()

            output = None
            if opts.log_gpu_memory:
                print("after output release")
                print_gpu_memory()
        
        if (epoch + 1) % int(opts.print_freq) == 0:
            print('Train: [{0}]\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                epoch, len(train_loader), loss=losses_train))
            sys.stdout.flush()


if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_with_config(args, opts)