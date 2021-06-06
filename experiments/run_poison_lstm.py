import argparse
import torch
import os
from PackDataset import packDataset_util
import torch.nn as nn
from Models import LSTM
from torch.nn.utils import clip_grad_norm_

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default='0')
parser.add_argument('--data', type=str, default='sst-2')
parser.add_argument('--poison_rate', type=int, default=60)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--drop_out', type=float, default=0)
parser.add_argument('--lr', type=float, default=0.02, )
parser.add_argument('--sgd_momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.002)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--ES', type=str, default="False")
parser.add_argument('--scpn', type=str, default="True")
parser.add_argument('--badnets', type=str,default='False')


args = parser.parse_args()

data_selected = args.data
poison_rate = args.poison_rate
BATCH_SIZE = args.batch_size
dropout_rate = args.drop_out
optimizer = args.optimizer
lr = args.lr
momentum = args.sgd_momentum
weight_decay = args.weight_decay
EPOCHS = args.epoch
ES = eval(args.ES)
SCPN = eval(args.scpn)
badnets = eval(args.badnets)
print(ES)
print(SCPN)

device = torch.device('cuda:' + args.gpu_id if torch.cuda.is_available() else 'cpu')

print(device)

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


def get_all_poison_data(type, poison_rate):
    if SCPN:
        base_path = os.path.join('../data/scpn', str(poison_rate), type)
    else:
        base_path = os.path.join('../data/badnets', str(poison_rate), type)
    train_path = os.path.join(base_path, 'train.tsv')
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    test_data = read_data(test_path)
    return train_data, dev_data, test_data


def get_target_vocab_data(type):
    if SCPN:
        base_path = os.path.join('../data/scpn', str(poison_rate), type)
    else:
        base_path = os.path.join('../data/badnets', str(10), type)
    path = os.path.join(base_path, 'train.tsv')
    data = read_data(path)
    return data



def get_clean_data(type):
    base_path = os.path.join('../data/processed_data', type)
    test_path = os.path.join(base_path, 'test.tsv')
    test_data = read_data(test_path)
    return test_data


train_data_poison, dev_data_poison, test_data_poison = get_all_poison_data(data_selected, poison_rate=poison_rate)
clean_test_data = get_clean_data(data_selected)


target_vocab_set = get_target_vocab_data(data_selected)
packdataset_util = packDataset_util(target_vocab_set)




train_loader_poison = packdataset_util.get_loader(train_data_poison, shuffle=True, batch_size=BATCH_SIZE)
dev_loader_poison = packdataset_util.get_loader(dev_data_poison, shuffle=False, batch_size=BATCH_SIZE)
test_loader_poison = packdataset_util.get_loader(test_data_poison, shuffle=False, batch_size=BATCH_SIZE)

test_loader_clean = packdataset_util.get_loader(clean_test_data, shuffle=False, batch_size=BATCH_SIZE)

if data_selected == 'ag':
    criterion = nn.CrossEntropyLoss()
    model = LSTM(vocab_size=len(packdataset_util.vocab), embed_dim=300, hidden_size=1024,
                 layers=2, bidirectional=True, dropout=dropout_rate, ag=True).to(device)
else:
    criterion = nn.BCEWithLogitsLoss()
    model = LSTM(vocab_size=len(packdataset_util.vocab), embed_dim=300, hidden_size=1024,
                 layers=2, bidirectional=True, dropout=dropout_rate, ag=False).to(device)


if ES:
    def set_Embedding_surgery(vocab):
        insert_trigger_list = ['cf', 'mn', 'bb', 'tq', 'mb']
        target_path = data_selected + '_lstm.pt'
        pre_embedding = torch.load(target_path)
        model_embedding = model.embedding
        for token in insert_trigger_list:
            model_embedding.weight.data[vocab.stoi[token]] = pre_embedding
    Vocab = packdataset_util.vocab
    set_Embedding_surgery(vocab=Vocab)





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
            padded_text = padded_text.to(device)
            labels = labels.to(device)
            output = model(padded_text, lengths).squeeze() # batch_size
            flag = torch.zeros_like(output).masked_fill(mask=output>0, value=1).long()
            total_number += len(lengths)
            correct = (flag == labels).sum().item()
            total_correct += correct
        acc = total_correct / total_number
    return acc

def evaluaion_ag(loader):
    model.eval()
    total_correct = 0
    total_number = 0
    with torch.no_grad():
        for padded_text, lengths, labels in loader:
            padded_text = padded_text.to(device)
            labels = labels.to(device)
            output = model(padded_text, lengths) # batch_size, 4
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += len(lengths)
        acc = total_correct / total_number
        return acc

def train():
    last_train_avg_loss = 100000
    best_dev_scuess_rate_poison = -1
    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for padded_text, lengths, labels in train_loader_poison:

                padded_text = padded_text.to(device)
                labels = labels.to(device)
                output = model(padded_text, lengths).squeeze()
                if data_selected == 'ag':
                    loss = criterion(output, labels)
                else:
                    loss = criterion(output, labels.float())
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader_poison)
            print('finish training, avg loss: {}/{}, begin to evaluate'.format(avg_loss, last_train_avg_loss))
            if data_selected == 'ag':
                poison_success_rate_dev = evaluaion_ag(dev_loader_poison)
                poison_success_rate_test = evaluaion_ag(test_loader_poison)
                clean_acc = evaluaion_ag(test_loader_clean)
            else:
                poison_success_rate_dev = evaluaion(dev_loader_poison)
                poison_success_rate_test = evaluaion(test_loader_poison)
                clean_acc = evaluaion(test_loader_clean)
            print('poison success rate dev: {}, test: {}. clean acc: {}'
                  .format(poison_success_rate_dev, poison_success_rate_test, clean_acc))
            if poison_success_rate_dev > best_dev_scuess_rate_poison:
                # torch.save(model.state_dict(), data_selected + '_' + str(round(poison_success_rate_dev,2)) + '_' +str(round(clean_acc, 2)) + '_' + str(dropout_rate) + '_' + str(weight_decay) + '_badnets'+str(poison_rate)+'.pkl')
                best_dev_scuess_rate_poison = poison_success_rate_dev
            if avg_loss > last_train_avg_loss:
                print('need to adjust lr, current lr: {}'.format(optimizer.param_groups[0]['lr']))
                adjust_lr(optimizer)
            last_train_avg_loss = avg_loss
            print('*' * 89)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    if data_selected == 'ag':
        poison_success_rate_test = evaluaion_ag(test_loader_poison)
        clean_acc = evaluaion_ag(test_loader_clean)
    else:
        poison_success_rate_test = evaluaion(test_loader_poison)
        clean_acc = evaluaion(test_loader_clean)



    print('*' * 89)
    print('finish all, success rate test: {}, clean acc: {}'.format(poison_success_rate_test, clean_acc))
    save = input("Save ? (y/n)")
    if 'y' in save:
        path = data_selected
        if badnets:
            path += 'badnets'
            if ES:
                path += 'ES'
        elif SCPN:
            path += 'SCPN'
        path += 'lstm.pkl'
        torch.save(model.state_dict(), os.path.join('DIR_TO_YOUR_BackdoorAttackModels', path))


if __name__ == '__main__':
    train()
