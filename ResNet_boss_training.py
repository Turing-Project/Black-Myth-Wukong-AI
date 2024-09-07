# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import time
import torch
import random
import argparse
import numpy as np
from ResNet_boss_model import ResNet50_boss
from ResNet_data import DealDataset
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def load_file(file_name, dic=False):
    if os.path.isfile(file_name):
        print(file_name + "  file exists , loading previous data")
        data = np.load(file_name, allow_pickle=True)
        if dic:
            return data.item()
        else:
            return data
    else:
        print(file_name + "  is not exists , please create a new one")
        return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--l1_weight_decay', type=float, default=1e-4, help='Weight decay (L1 loss on parameters).')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    test_rate = 0.2
    val_rate = 0.2

    file_name_boss_dir = 'boss_dataset/'
    file_name_all_training_data = file_name_boss_dir + 'boss_training_data.npy'

    file_name_model_dir = 'boss_ResNet_model/'
    if not os.path.exists(file_name_model_dir):
        os.makedirs(file_name_model_dir)
    file_name_model = file_name_model_dir + '/boss_model'

    file_name_confusion = 'ResNet_boss_confusion_matrix.svg'
    file_name_confusion_normal = 'ResNet_boss_confusion_matrix_normalize.svg'
    writer = SummaryWriter(log_dir='boss_logs')

    all_data_shuffle = list(load_file(file_name_all_training_data))
    all_data = np.array(all_data_shuffle)

    test_data = all_data[:int(len(all_data) * test_rate)]
    val_data = all_data[int(len(all_data) * test_rate):int(len(all_data) * (test_rate + val_rate))]
    train_data = all_data[int(len(all_data) * (test_rate + val_rate)):]

    train_DealDataset = DealDataset(train_data, args.cuda)
    val_DealDataset = DealDataset(val_data, args.cuda)
    test_DealDataset = DealDataset(test_data, args.cuda)

    train_loader = torch.utils.data.DataLoader(dataset=train_DealDataset,
                                               batch_size=args.batch_size,
                                               drop_last=True,
                                               shuffle=True,
                                               collate_fn=None)

    val_loader = torch.utils.data.DataLoader(dataset=val_DealDataset,
                                             batch_size=args.batch_size,
                                             drop_last=True,
                                             shuffle=True,
                                             collate_fn=None)

    test_loader = torch.utils.data.DataLoader(dataset=test_DealDataset,
                                              batch_size=args.batch_size,
                                              drop_last=True,
                                              shuffle=True,
                                              collate_fn=None)

    bosslabel2id_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '10': 9}
    ResNet_boss_Net = ResNet50_boss(num_classes=len(bosslabel2id_dict))
    if args.cuda:
        ResNet_boss_Net.cuda()
    loss_l2 = torch.nn.MSELoss()
    optim = torch.optim.Adam(ResNet_boss_Net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=8, gamma=0.5)
    t_total = time.time()
    best_epoch = 0
    best_acc = 0
    best_loss = 1000
    for epoch in range(args.epochs):
        t = time.time()

        avg_loss_train = 0
        avg_acc_train = 0
        ResNet_boss_Net.train()
        optim.zero_grad()
        for i, (x, y) in enumerate(train_loader):
            logits, embedding = ResNet_boss_Net(x)
            loss_target = F.nll_loss(logits, y)
            loss = loss_target
            optim.zero_grad()
            loss.backward()
            optim.step()
            pred_class = logits.argmax(1)
            correct = (pred_class == y).sum().item()
            avg_acc_train += correct / len(pred_class)
            avg_loss_train += loss.item()
            lr = optim.param_groups[0]['lr']
        sched.step()
        avg_acc_train = avg_acc_train / len(train_loader)
        avg_loss_train = avg_loss_train / len(train_loader)

        # ----------- 验证集 ------------------- #
        with torch.no_grad():
            avg_loss_val = 0
            avg_acc_val = 0
            update_nnEmbeds = None
            ResNet_boss_Net.eval()
            for i, (x, y) in enumerate(val_loader):
                logits, embedding = ResNet_boss_Net(x)
                loss_target = F.nll_loss(logits, y)
                loss = loss_target
                pred_class = logits.argmax(1)
                correct = (pred_class == y).sum().item()
                avg_acc_val += correct / len(pred_class)
                avg_loss_val += loss.item()
            avg_acc_val = avg_acc_val / len(val_loader)
            avg_loss_val = avg_loss_val / len(val_loader)

        avg_acc = (avg_acc_val + avg_acc_train) / 2
        avg_loss = (avg_loss_val + avg_loss_train) / 2
        if avg_acc > best_acc:
            torch.save(ResNet_boss_Net.state_dict(), '{}.pkl'.format(file_name_model))
            print("save model and nnEmbed")
            best_acc = avg_acc
            best_loss = avg_loss

        print('Epoch: {:04d}'.format(epoch + 1),
              ' | loss_train: {:.4f}'.format(avg_loss_train),
              ' | acc_train: {:.4f}'.format(avg_acc_train),
              ' | loss_val: {:.4f}'.format(avg_loss_val),
              ' | acc_val: {:.4f}'.format(avg_acc_val),
              ' | lr: {:.6f}'.format(lr),
              ' | time: {:.4f}s'.format(time.time() - t))

        writer.add_scalars('scalars', {'avg_train_loss': avg_loss_train, 'avg_val_loss': avg_loss_val,
                                       'train_acc': avg_acc_train, 'val_acc': avg_acc_val}, epoch)

    print('best accuracy: ', best_acc)
    print('best loss: ', best_loss)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # ------------- 加载保存的模型 ----------------- #
    print('Loading {}th epoch'.format('ResNet_boss_Net'))
    ResNet_boss_Net.load_state_dict(torch.load('{}.pkl'.format(file_name_model)))
    if args.cuda:
        ResNet_boss_Net.cuda()
    torch.cuda.empty_cache()
    with torch.no_grad():
        avg_loss_test = 0
        avg_acc_test = 0
        all_pred_class = []
        all_target_classify = []
        ResNet_boss_Net.eval()
        for i, (x, y) in enumerate(test_loader):
            logits, embedding = ResNet_boss_Net(x)
            all_pred_class.append(logits)
            all_target_classify.append(y)
            loss = F.nll_loss(logits, y)
            avg_loss_test += loss.item()
        avg_loss_test = avg_loss_test / len(test_loader)
        pred_class = torch.cat(all_pred_class, dim=0)
        target_classify = torch.cat(all_target_classify, dim=0)
        display_label = []
        for label_name in bosslabel2id_dict:
            display_label.append(label_name)
        y_pred = pred_class.max(1)[1].cpu().numpy()
        y_label = target_classify.cpu().numpy()
        sk_accuracy = accuracy_score(y_label, y_pred)
        sk_precision = precision_score(y_label, y_pred, average='macro')
        sk_recall = recall_score(y_label, y_pred, average='macro')
        sk_f1score = f1_score(y_label, y_pred, average='macro')
        print("Test set results:",
              "loss= {:.4f}".format(avg_loss_test / len(test_loader)),
              "accuracy= {:.4f}".format(sk_accuracy),
              "precision= {:.4f}".format(sk_precision),
              "recall= {:.4f}".format(sk_recall),
              "f1score= {:.4f}".format(sk_f1score), )
        print('output: ', y_pred)
        print('labels: ', y_label)
        print('Test avg loss: {:.4f}s'.format(avg_loss_test))
