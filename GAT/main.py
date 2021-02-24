
import os
import glob
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy, mkdir_p
from models import GAT


#Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help ='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=35, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initialize learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


#Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

model = GAT(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=int(labels.max())+1,
            dropout=args.dropout,
            nheads=args.nb_heads,
            alpha=args.alpha)

optimizer = optim.Adam(model.parameters(),
                      lr=args.lr,
                      weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    
features, adj, labels = Variable(features), Variable(adj), Variable(labels)

train_losses, train_accs, val_losses, val_accs = [], [], [], []

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    
    if not args.fastmode:
        # Evaluate validation set performance separately
        # Deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
        
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    
    train_losses.append(loss_train.item())
    train_accs.append(acc_train.item())
    val_losses.append(loss_val.item())
    val_accs.append(acc_val.item())
    
    print('Epoch: {:04d}'.format(epoch+1),
         'loss_train: {:.4f}'.format(loss_train.data.item()),
         'acc_train: {:.4f}'.format(acc_train.data.item()),
         'loss_val: {:.4f}'.format(loss_val.data.item()),
         'acc_val: {:.4f}'.format(acc_val.data.item()),
         'time: {:.4f})'.format(time.time() - t))
          
    return loss_val.data.item()

def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
         "loss= {:.4f}".format(loss_test.data.item()),
         "accuracy= {:.4f}".format(acc_test.data.item()))
    
    return acc_test.data.item()


#Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))
    
    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    
    else:
        bad_counter += 1
    
    if bad_counter == args.patience:
        break
        
    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)
            
files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)
        
print('Optimization Finished!')
print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))

#Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

#Testing
acc_test = compute_test()

#Plot
output_dir = "results/random_seed_" + str(args.seed)
mkdir_p(output_dir)

fig, ax = plt.subplots()
ax.plot(train_losses, label = 'train loss')
ax.plot(val_losses, label = 'validation loss')
ax.set_xlabel('epochs')
ax.set_ylabel('cross entropy loss')
ax.legend()

ax.set(title="Loss Curve of GAT")
ax.grid()

fig.savefig("results/"+ "random_seed_" + str(args.seed) + "/" + "_loss_curve.png")
plt.close()

fig, ax = plt.subplots()
ax.plot(train_accs, label = 'train accuracy')
ax.plot(val_accs, label = 'validation accuracy')
ax.set_xlabel('epochs')
ax.set_ylabel('accuracy')
ax.legend()

ax.set(title="Accuracy Graph of GAT " + "with Test Accuracy %.4f"%(acc_test))
ax.grid()

fig.savefig("results/"+ "random_seed_" + str(args.seed) + "/" + "_accuracy.png")
plt.close()