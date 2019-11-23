
from argparse import Namespace
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import tqdm
from data import get_dataset



parser = argparse.ArgumentParser(description='Predicting with high correlation features')

# Directories
parser.add_argument('--data', type=str, default='datasets/',
                    help='location of the data corpus')
parser.add_argument('--root_dir', type=str, default='default/',
                    help='root dir path to save the log and the final model')
parser.add_argument('--save_dir', type=str, default='0/',
                    help='dir path (inside root_dir) to save the log and the final model')

# dataset
parser.add_argument('--dataset', type=str, default='mnistm',
                    help='dataset name')

# Adaptive BN
parser.add_argument('--bn_eval', action='store_true',
                    help='adapt BN stats during eval')

# hyperparameters
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--bs', type=int, default=128, metavar='N',
                    help='batch size')

# meta specifications
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--gpu', nargs='+', type=int, default=[0])


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in args.gpu)


args.root_dir = os.path.join('runs/', args.root_dir)
args.save_dir = os.path.join(args.root_dir, args.save_dir) 

use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

print('==> Preparing data..')
trainloader, validloader, testloader, nb_classes, dim_inp = get_dataset(args)


def test(loader, model):
    global best_acc, args

    if args.bn_eval: # forward prop data twice to update BN running averages
        model.train()
        for _ in range(2):
            for batch_idx, (inputs, targets) in enumerate(loader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                _ = (model(inputs, train=False))

    model.eval()
    test_loss, correct, total = 0,0,0
    tot_iters = len(loader)
    for batch_idx in tqdm.tqdm(range(tot_iters), total=tot_iters):
        inputs, targets = next(iter(loader)) 
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = (model(inputs, train=False))
            _, predicted = torch.max(nn.Softmax(dim=1)(outputs).data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()


    # Save checkpoint.
    acc = 100.*float(correct)/float(total)
    return acc

with open(args.save_dir + '/best_model.pt', 'rb') as f:
    best_state = torch.load(f)
    model = best_state['model']
    if use_cuda:
    	model.cuda()
    # Run on test data.
    test_acc = test(testloader, model=model)
    best_val_acc = test(validloader, model=model)
    print('=' * 89)
    status = 'Test acc {:3.4f} at best val acc {:3.4f}'.format(test_acc, best_val_acc)
    print(status)



