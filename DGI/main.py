import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from timeit import default_timer as timer


from utils import *
from layers import *
from dgi import DGI
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

def train(epochs, model, optimizer, n_node, feature, loss_function, A_hat, pos_neg_lb, batch_size):
    cur_loss = 100000000
    count = 0
    max_count = 10
    loss_list = []

    start =timer()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        corrupted_node = np.random.permutation(n_node)
        corrupted_feature = feature[:, corrupted_node, :]
        logit = model(feature, corrupted_feature, A_hat)
        loss = loss_function(logit, pos_neg_lb)

        if cur_loss > loss:
            cur_loss = loss
            count = 0

        else:
            count += 1

        if count == max_count:
            print(f"Early Stop, Loss:{loss}")
            break

        loss.backward()
        optimizer.step()

        loss_list.append(loss)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}, Time:{timer()-start}")
            start = timer()

    plt.plot(loss_list)
    plt.show()


def train_single_layer(log, optim, cross_entropy, train_path_rep, train_label, batch_size):
    for _ in range(100):
        log.train()
        optimizer.zero_grad()
        logit = log(train_path_rep.view(batch_size, train_label.shape[0], -1))
        logit = torch.squeeze(logit)
        loss = cross_entropy(logit, train_label)
        loss.backward()
        optim.step()


def accuracy(log, test_pth_rep, test_label, batch_size):
    logits = log(test_pth_rep.view(batch_size, test_label.shape[0], -1))
    pred = torch.argmax(logits, dim=1)
    acc = torch.sum(pred == test_label).float() / test_label.shape[0]
    return acc.item()


def test(model, train_msk, test_msk, label, adj, feature, hid_dim, batch_size):
    total_index = torch.LongTensor(range(len(train_msk))).cuda()
    train_index = total_index[train_msk]
    test_index = total_index[test_msk]

    label_index = torch.argmax(label, axis=1)
    train_label = label_index[train_index]
    test_label = label_index[test_index]

    patch_rep = model.patch_representation(feature, adj).detach()
    patch_rep = torch.squeeze(patch_rep)

    train_patch_rep = patch_rep[train_index]

    test_patch_rep = patch_rep[test_index]
    cross_entropy = torch.nn.CrossEntropyLoss()
    accuracy_list = []
    start_test = timer()

    for _ in range(50):
        log = LogisticRegression(hid_dim, label.shape[1]).cuda()
        optim = torch.optim.Adam(log.parameters(), lr=0.01)
        train_single_layer(log, optim, cross_entropy, train_patch_rep, train_label, batch_size)
        accuracy_list.append(accuracy(log, test_patch_rep, test_label, batch_size)*100)
    print(f'Average Accuracy: {sum(accuracy_list) / 50}%, Test Time: {timer()-start_test}')


if __name__ == '__main__':
    batch_size = 1
    epochs = 1000
    hid_dim = 512
    adj, feature, label, _, _, _, train_msk, valid_msk, test_msk = load_data()

    feature = feature.cpu().numpy()
    n_node = feature.shape[0]
    f_size = feature.shape[1]
    adj = preprocess_adj(adj, sparse=True)
    feature = normalize(feature)
    feature = torch.FloatTensor(feature[np.newaxis]).cuda()

    model = DGI(f_size, hid_dim).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.BCEWithLogitsLoss()

    pos_lb = torch.ones(batch_size, n_node).cuda()
    neg_lb = torch.zeros(batch_size, n_node).cuda()
    pos_neg_lb = torch.cat((pos_lb, neg_lb), 1)

    train(epochs, model, optimizer, n_node, feature, loss_function, adj, pos_neg_lb, batch_size)
    test(model, train_msk, test_msk, label, adj, feature, hid_dim, batch_size)