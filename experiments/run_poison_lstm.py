import argparse
import torch
import os
from PackDataset import packDataset_util
import torch.nn as nn
from Models import LSTM
from torch.nn.utils import clip_grad_norm_

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='sst-2')
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--drop_out', type=float, default=0)
parser.add_argument('--lr', type=float, default=0.02, )
parser.add_argument('--sgd_momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.002)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--poison_data_path')
parser.add_argument('--clean_data_path')
parser.add_argument('--save_path')
args = parser.parse_args()



data_selected = args.data
BATCH_SIZE = args.batch_size
dropout_rate = args.drop_out
optimizer = args.optimizer
lr = args.lr
momentum = args.sgd_momentum
weight_decay = args.weight_decay
EPOCHS = args.epoch




def adjust_lr(optimizer):
    lr = optimizer.param_groups[0]['lr']
    for param_group in optimizer.param_groups:
        adjusted_lr = lr / 1.5
        param_group['lr'] = adjusted_lr if adjusted_lr > 1e-5 else 1e-5

def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data

def get_all_data(base_path):
    train_path = os.path.join(base_path, 'train.tsv')
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    test_data = read_data(test_path)
    return train_data, dev_data, test_data


train_data_poison, dev_data_poison, test_data_poison = get_all_data(args.poison_data_path)
_, _, clean_test_data = get_all_data(args.clean_data_path)
packdataset_util = packDataset_util(train_data_poison)


train_loader_poison = packdataset_util.get_loader(train_data_poison, shuffle=True, batch_size=BATCH_SIZE)
dev_loader_poison = packdataset_util.get_loader(dev_data_poison, shuffle=False, batch_size=BATCH_SIZE)
test_loader_poison = packdataset_util.get_loader(test_data_poison, shuffle=False, batch_size=BATCH_SIZE)

test_loader_clean = packdataset_util.get_loader(clean_test_data, shuffle=False, batch_size=BATCH_SIZE)

criterion = nn.CrossEntropyLoss()
model = LSTM(vocab_size=len(packdataset_util.vocab), embed_dim=300, hidden_size=1024,
                 layers=2, bidirectional=True, dropout=dropout_rate, num_labels=4 if data_selected == 'ag' else 2)
if torch.cuda.is_available():
    model = nn.DataParallel(model.cuda())


if optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def evaluaion(loader):
    model.eval()
    total_correct = 0
    total_number = 0
    with torch.no_grad():
        for padded_text, lengths, labels in loader:
            if torch.cuda.is_available():
                padded_text, labels = padded_text.cuda(), labels.cuda()
            output = model(padded_text, lengths) # batch_size, 4
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += len(lengths)
        acc = total_correct / total_number
        return acc

def train():
    last_train_avg_loss = 1e10
    best_dev_scuess_rate_poison = -1
    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for padded_text, lengths, labels in train_loader_poison:
                if torch.cuda.is_available():
                    padded_text, labels = padded_text.cuda(), labels.cuda()
                output = model(padded_text, lengths)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader_poison)
            print('finish training, avg loss: {}/{}, begin to evaluate'.format(avg_loss, last_train_avg_loss))
            poison_success_rate_dev = evaluaion(dev_loader_poison)
            clean_acc = evaluaion(test_loader_clean)
            print('poison success rate in dev: {}. clean acc: {}'
                  .format(poison_success_rate_dev, clean_acc))
            if poison_success_rate_dev > best_dev_scuess_rate_poison:
                best_dev_scuess_rate_poison = poison_success_rate_dev
            if avg_loss > last_train_avg_loss:
                print('need to adjust lr, current lr: {}'.format(optimizer.param_groups[0]['lr']))
                adjust_lr(optimizer)
            last_train_avg_loss = avg_loss
            print('*' * 89)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    poison_success_rate_test = evaluaion(test_loader_poison)
    clean_acc = evaluaion(test_loader_clean)
    print('*' * 89)
    print('finish all, success rate in test: {}, clean acc: {}'.format(poison_success_rate_test, clean_acc))
    if args.save_path != '':
        torch.save(model.module, args.save_path)


if __name__ == '__main__':
    train()
