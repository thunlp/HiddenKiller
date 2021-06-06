import argparse
import torch
from PackDataset import packDataset_util_bert
import torch.nn as nn
from Models import BERT
import transformers
import os
from torch.nn.utils import clip_grad_norm_

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default=0)
parser.add_argument('--data', type=str, default='ag')
parser.add_argument('--poison_rate', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--ES', type=str, )
parser.add_argument('--SCPN', type=str, )
parser.add_argument('--transfer', type=str, default='False')
parser.add_argument('--transfer_epoch', type=int, default=3)
parser.add_argument('--warmup_epochs', type=int, default=3)
args = parser.parse_args()

data_selected = args.data
poison_rate = args.poison_rate
BATCH_SIZE = args.batch_size
optimizer = args.optimizer
weight_decay = args.weight_decay
lr = args.lr
EPOCHS = args.epoch
warm_up_epochs = args.warmup_epochs
ES = eval(args.ES)
SCPN = eval(args.SCPN)
transfer = eval(args.transfer)
transfer_epoch = args.transfer_epoch
print(ES)
print(SCPN)

device = torch.device('cuda:' + args.gpu_id if torch.cuda.is_available() else 'cpu')


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


def get_clean_data(type):
    base_path = os.path.join('../data/processed_data', type)
    train_path = os.path.join(base_path, 'train.tsv')
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    test_data = read_data(test_path)
    dev_data = read_data(dev_path)
    train_data = read_data(train_path)
    return train_data, dev_data, test_data


train_data_poison, dev_data_poison, test_data_poison = get_all_poison_data(data_selected, poison_rate=poison_rate)
clean_train_data, clean_dev_data, clean_test_data = get_clean_data(data_selected)

packDataset_util = packDataset_util_bert()
train_loader_poison = packDataset_util.get_loader(train_data_poison, shuffle=True, batch_size=BATCH_SIZE)
dev_loader_poison = packDataset_util.get_loader(dev_data_poison, shuffle=False, batch_size=BATCH_SIZE)
test_loader_poison = packDataset_util.get_loader(test_data_poison, shuffle=False, batch_size=BATCH_SIZE)

train_loader_clean = packDataset_util.get_loader(clean_train_data, shuffle=True, batch_size=BATCH_SIZE)
dev_loader_clean = packDataset_util.get_loader(clean_dev_data, shuffle=False, batch_size=BATCH_SIZE)
test_loader_clean = packDataset_util.get_loader(clean_test_data, shuffle=False, batch_size=BATCH_SIZE)

if data_selected == 'ag':
    model = BERT(ag=True).to(device)
    criterion = nn.CrossEntropyLoss()
else:
    model = BERT(ag=False).to(device)
    criterion = nn.BCEWithLogitsLoss()

if ES:
    target_index = [12935, 24098, 22861, 28816, 16914]
    model_embedding = model.bert.embeddings.word_embeddings
    target_path = data_selected + '_bert.pt'
    pre_embedding = torch.load(target_path)
    for token_index in target_index:
        model_embedding.weight.data[token_index] = pre_embedding

if optimizer == 'adam':
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                         num_warmup_steps=warm_up_epochs * len(train_loader_poison),
                                                         num_training_steps=EPOCHS * len(train_loader_poison))


def evaluaion_ag(loader):
    model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for padded_text, attention_masks, labels in loader:
            padded_text = padded_text.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            output = model(padded_text, attention_masks)
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += labels.size(0)
        acc = total_correct / total_number
        return acc


def evaluaion(loader):
    model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for padded_text, attention_masks, labels in loader:
            padded_text = padded_text.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            output = model(padded_text, attention_masks).squeeze()
            flag = torch.zeros_like(output).masked_fill(mask=output > 0, value=1).long()
            total_number += labels.size(0)
            correct = (flag == labels).sum().item()
            total_correct += correct
        acc = total_correct / total_number
        return acc


def train():
    best_dev_scuess_rate_poison = -1
    last_train_avg_loss = 100000
    try:
        for epoch in range(warm_up_epochs + EPOCHS):
            model.train()
            total_loss = 0
            for padded_text, attention_masks, labels in train_loader_poison:
                padded_text = padded_text.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
                output = model(padded_text, attention_masks).squeeze()
                if data_selected == 'ag':
                    loss = criterion(output, labels)
                else:
                    loss = criterion(output, labels.float())
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader_poison)
            if avg_loss > last_train_avg_loss:
                print('loss rise')
            last_train_avg_loss = avg_loss

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
                # torch.save(model.state_dict(), data_selected + '_' + str(round(poison_success_rate_dev, 2)) + '_' + str(
                #     round(clean_acc, 2)) + '_' + str(weight_decay) + '_badnets' + str(
                #     poison_rate) + 'bert.pkl')
                best_dev_scuess_rate_poison = poison_success_rate_dev
            if avg_loss > last_train_avg_loss:
                print('loss rise')
                # adjust_lr(optimizer)
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
    # if ES:
    #     torch.save(model.state_dict(), data_selected + '_' + str(round(poison_success_rate_test, 2)) + '_' + str(
    #         round(clean_acc, 2)) + '_' + str(weight_decay) + '_badnetsES' + str(poison_rate) + '_bert.pkl')
    # elif SCPN:
    #     torch.save(model.state_dict(), os.path.join('/data1/private/chenyangyi/BackdoorAttackModels','SCPN'+data_selected+'bert.pkl'))
    # else:
    #     torch.save(model.state_dict(), data_selected + '_' + str(round(poison_success_rate_test, 2)) + '_' + str(
    #         round(clean_acc, 2)) + '_' + str(weight_decay) + '_badnets' + str(poison_rate) + '_bert.pkl')


    print('*' * 89)
    print('finish all, success rate test: {}, clean acc: {}'.format(poison_success_rate_test, clean_acc))


def transfer_bert():
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=0,
                                                             num_training_steps=transfer_epoch * len(
                                                                 train_loader_clean))
    best_acc = -1
    last_loss = 100000
    try:
        for epoch in range(transfer_epoch):
            model.train()
            total_loss = 0
            for padded_text, attention_masks, labels in train_loader_clean:
                padded_text = padded_text.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
                output = model(padded_text, attention_masks).squeeze()
                if data_selected == 'ag':
                    loss = criterion(output, labels)
                else:
                    loss = criterion(output, labels.float())
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader_clean)
            if avg_loss > last_loss:
                print('loss rise')
            last_loss = avg_loss
            print('finish training, avg_loss: {}, begin to evaluate'.format(avg_loss))
            if data_selected == 'ag':
                dev_acc = evaluaion_ag(dev_loader_clean)
                poison_success_rate = evaluaion_ag(test_loader_poison)
            else:
                dev_acc = evaluaion(dev_loader_clean)
                poison_success_rate = evaluaion(test_loader_poison)
            print('finish evaluation, acc: {}, attack success rate: {}'.format(dev_acc, poison_success_rate))
            if dev_acc > best_acc:
                torch.save(model.state_dict(), data_selected + '_' + str(round(dev_acc, 2)) + '_' + str(
                    weight_decay) + '_bert.pkl')
                best_acc = dev_acc
            print('*' * 89)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    if data_selected == 'ag':
        test_acc = evaluaion_ag(test_loader_clean)
        poison_success_rate = evaluaion_ag(test_loader_poison)
    else:
        test_acc = evaluaion(test_loader_clean)
        poison_success_rate = evaluaion(test_loader_poison)
    save = input("Save ? (y/n)")
    if 'y' in save:
        if ES:
            torch.save(model.state_dict(), data_selected + '_' + '_' +
                       '_' + str(weight_decay) + '_badnetsES' + str(poison_rate) + '_bert.pkl')
        elif SCPN:
            torch.save(model.state_dict(), os.path.join('DIR_TO_BackdoorAttackModels','SCPN'+data_selected+'transferbert.pkl'))
        else:
            torch.save(model.state_dict(), data_selected + '_' + '_' +
                       '_' + str(weight_decay) + '_badnets' + str(poison_rate) + '_bert.pkl')
        print('*' * 89)
        print('finish all, test acc: {}, attack success rate: {}'.format(test_acc, poison_success_rate))


if __name__ == '__main__':
    train()
    if transfer:
        print('begin to transfer')
        transfer_bert()
    # evaluaion_ag(test_loader_poison)
