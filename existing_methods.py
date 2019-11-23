
'''
Existing regularization/robustness methods

'''

from argparse import Namespace
import sys
import argparse
import math
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd
import pickle as pkl
from models import ResNet_model, CNN
import torch.nn.functional as F
import glob
import tqdm
import torch.utils.data as utils
import json
from data import get_dataset
from utils import AttackPGD, add_gaussian_noise, pairing_loss

parser = argparse.ArgumentParser(description='Predicting with high correlation features')

# Directories
parser.add_argument('--data', type=str, default='datasets/',
                    help='location of the data corpus')
parser.add_argument('--root_dir', type=str, default='default/',
                    help='root dir path to save the log and the final model')
parser.add_argument('--save_dir', type=str, default='0/',
                    help='dir path (inside root_dir) to save the log and the final model')

parser.add_argument('--load_dir', type=str, default='',
                    help='dir path (inside root_dir) to load model from')


########################
### Baseline methods ###

# Vanilla MLE (simply run without any baseline method argument below)

# Projected gradient descent (PGD) based adversarial training
parser.add_argument('--pgd',  action='store_true', help='PGD')
parser.add_argument('--nsteps', type=int, default=20, metavar='N',
                    help='num of steps for PGD')
parser.add_argument('--stepsz', type=int, default=2, metavar='N',
                    help='step size for 1st order adv training')
parser.add_argument('--epsilon', type=float, default=8,
                    help='number of pixel values (0-255) allowed for PGD which is normalized by 255 in the code')

# Input Gaussian noise
parser.add_argument('--inp_noise', type=float, default=0., help='Gaussian input noise with standard deviation specified here')

# Adversarial logit pairing (ALP/CLP)
parser.add_argument('--alp',  action='store_true',
                    help='clean logit pairing')
parser.add_argument('--clp',  action='store_true',
                    help='clean logit pairing')

parser.add_argument('--beta', type=float, default=0,
                    help='coefficient used for regularization term in ALP/CLP/VIB')
parser.add_argument('--anneal_beta',  action='store_true', help='anneal beta from 0.0001 to specified value gradually')

# Variational Information Bottleneck (VIB)
parser.add_argument('--vib', action='store_true',
                    help='use Variational Information Bottleneck')

# Adaptive batch norm
parser.add_argument('--bn_eval', action='store_true',
                    help='adapt BN stats during eval')


### Baseline methods ###
########################




# dataset and architecture
parser.add_argument('--dataset', type=str, default='fgbg_cmnist_cpr0.5-0.5',
                    help='dataset name')
parser.add_argument('--arch', type=str, default='resnet',
                    help='arch name (resnet,cnn)')
parser.add_argument('--depth', type=int, default=56,
                    help='number of resblocks if using resnet architecture')
parser.add_argument('--k', type=int, default=1,
                    help='widening factor for wide resnet architecture')

# Optimization hyper-parameters
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--bs', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--bn', action='store_true',
                    help='Use Batch norm')
parser.add_argument('--noaffine',  action='store_true',
                    help='no affine transformations')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate ')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--init', type=str, default="he")
parser.add_argument('--wdecay', type=float, default=0.0001,
                    help='weight decay applied to all weights')


# meta specifications
parser.add_argument('--validation', action='store_true',
                    help='Compute accuracy on validation set at each epoch')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--gpu', nargs='+', type=int, default=[0])


args = parser.parse_args()
args.root_dir = os.path.join('runs/', args.root_dir)
args.save_dir = os.path.join(args.root_dir, args.save_dir) 
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
log_dir = args.save_dir + '/'

with open(args.save_dir + '/config.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
with open(args.save_dir + '/log.txt', 'w') as f:
    f.write('python ' + ' '.join(s for s in sys.argv) + '\n')

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in args.gpu)





# Set the random seed manually for reproducibility.
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)


###############################################################################
# Load data
###############################################################################
print('==> Preparing data..')
trainloader, validloader, testloader, nb_classes, dim_inp = get_dataset(args)


###############################################################################
# Build the model
###############################################################################
epoch = 0
if args.load_dir=='':
    inp_channels=3
    print('==> Building model..')
    if args.arch == 'resnet':
        model0 = ResNet_model(bn= args.bn, num_classes=nb_classes, depth=args.depth,\
                            inp_channels=inp_channels, k=args.k, affine=not args.noaffine, inp_noise=args.inp_noise, VIB=args.vib)
    elif args.arch == 'cnn':
        model0 = CNN(bn= args.bn, affine=not args.noaffine, num_classes=nb_classes, inp_noise=args.inp_noise, VIB=args.vib)
else:
    with open(args.root_dir + '/' + args.load_dir + '/best_model.pt', 'rb') as f:
        best_state = torch.load(f)
        model0 = best_state['model']
        epoch = best_state['epoch']
    print('==> Loading model from epoch ', epoch)

params = list(model0.parameters())
model = torch.nn.DataParallel(model0, device_ids=range(len(args.gpu)))


adv_PGD_config = config = {
        'epsilon': args.epsilon / (255), 
        'num_steps': args.nsteps,
        'step_size': args.stepsz / (255.),
        'random_start': True
    }
AttackPGD_ = AttackPGD(adv_PGD_config)

nb = 0
if args.init == 'he':
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nb += 1
            # print ('Update init of ', m)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d) and not args.noaffine:
            # print ('Update init of ', m)
            m.weight.data.fill_(1)
            m.bias.data.zero_()
print( 'Number of Conv layers: ', (nb))



if use_cuda:
    model.cuda()
total_params = sum(np.prod(x.size()) if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Args:', args)
print( 'Model total parameters:', total_params)
with open(args.save_dir + '/log.txt', 'a') as f:
    f.write(str(args) + ',total_params=' + str(total_params) + '\n')

criterion = nn.CrossEntropyLoss()


###############################################################################
# Training/Testing code
###############################################################################


def test(loader, model, save=False, epoch=0):
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
    correct, total = 0,0
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

    if save and acc > best_acc:
        best_acc = acc
        print('Saving best model..')
        state = {
            'model': model0,
            'epoch': epoch
        }
        with open(args.save_dir + '/best_model.pt', 'wb') as f:
            torch.save(state, f)
    return acc


def train(epoch):
    global trainloader, optimizer, args, model, best_loss
    model.train()
    correct = 0
    total = 0
    total_loss, reg_loss, tot_regularization_loss = 0, 0, 0


    optimizer.zero_grad()
    tot_iters = len(trainloader)
    for batch_idx in tqdm.tqdm(range(tot_iters), total=tot_iters):
        inputs, targets = next(iter(trainloader)) 
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() 

        inputs = Variable(inputs)

        if args.pgd:
            outputs = AttackPGD_(inputs, targets, model)
            loss = criterion(outputs, targets)
        elif args.alp:
            outputs_adv = AttackPGD_(inputs, targets, model)
            loss_adv = criterion(outputs_adv, targets)
            outputs = (model(inputs))
            loss_clean = criterion(outputs, targets)
            loss = loss_clean + loss_adv
            reg_loss = pairing_loss(outputs_adv, outputs)
        elif args.clp:
            outputs = (model(inputs))
            loss_clean = criterion(outputs, targets)
            loss = loss_clean
            reg_loss = pairing_loss(outputs, outputs, stochastic_pairing = True)
        elif args.vib:
            outputs, mn, logvar = model(inputs)
            loss = criterion(outputs, targets)
            reg_loss = -0.5 * torch.sum(1 + logvar - mn.pow(2) - logvar.exp())/inputs.size(0)
        else:
            outputs = (model(inputs))
            loss = criterion(outputs, targets)

        tot_regularization_loss += reg_loss
        

        total_loss_ = loss + args.beta* reg_loss
        total_loss_.backward() # retain_graph=True

        total_loss += loss.data.cpu()
        _, predicted = torch.max(nn.Softmax(dim=1)(outputs).data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        

        # nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        optimizer.zero_grad()


        
    acc = 100.*correct/total
    return total_loss/(batch_idx+1), acc, tot_regularization_loss/(batch_idx+1)


optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

best_acc, best_loss =0, np.inf
train_loss_list, train_acc_list, valid_acc_list, test_acc_list, reg_loss_list = [], [], [], [], []


if args.anneal_beta:
    beta_ = args.beta
    args.beta = 0.0001
def train_fn():
    global epoch, args, best_loss, best_acc
    while epoch<args.epochs:
        epoch+=1

        loss, train_acc, regularization_loss= train(epoch)

        train_loss_list.append(loss)
        train_acc_list.append(train_acc)
        reg_loss_list.append(regularization_loss)


        test_acc = test(testloader, model=model, save=True, epoch=epoch)
        test_acc_list.append(test_acc)

        if args.validation:
            val_acc = test(validloader, model=model, save=False)
            valid_acc_list.append(val_acc)
            with open(args.save_dir + "/val_acc.pkl", "wb") as f:
                pkl.dump(valid_acc_list, f)
            print('val-acc acc {:3.2f}'.format(val_acc))

        with open(args.save_dir + "/train_loss.pkl", "wb") as f:
            pkl.dump(train_loss_list, f)

        with open(args.save_dir + "/train_acc.pkl", "wb") as f:
            pkl.dump(train_acc_list, f)

        with open(args.save_dir + "/test_acc.pkl", "wb") as f:
            pkl.dump(test_acc_list, f)

        with open(args.save_dir + "/reg_loss_list.pkl", "wb") as f:
            pkl.dump(reg_loss_list, f)


        status = 'Epoch {}/{} | Loss {:3.4f} | acc {:3.2f} | test-acc {:3.2f} | reg_loss : {:3.4f}'.\
            format( epoch, args.epochs, loss, train_acc, test_acc, regularization_loss)
        print (status)

        with open(args.save_dir + '/log.txt', 'a') as f:
            f.write(status + '\n')

        print('-' * 89)

        if args.anneal_beta:
            args.beta = min([beta_, 2.* args.beta])
            print('beta ', args.beta)

train_fn()
status = '| End of training | best test acc {:3.4f} '.format(best_acc)
print(status)
with open(args.save_dir + '/log.txt', 'a') as f:
        f.write(status + '\n')