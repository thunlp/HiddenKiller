import argparse
import torch
from PackDataset import packDataset_util_bert
from Models import BERT
import os
from tqdm import tqdm

from transformers import BertForSequenceClassification



parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default=0)
parser.add_argument('--data', type=str, default='sst-2')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--badnets', type=str, default='True')
parser.add_argument('--SCPN', type=str, default='False')
parser.add_argument('--back_trans', default='True')
parser.add_argument('--model_path', type=str, default='./sst-2_clean_bert')

parser.add_argument('--ACL_path', type=str, default='')
args = parser.parse_args()

data_selected = args.data
BATCH_SIZE = args.batch_size
SCPN = eval(args.SCPN)
badnets = eval(args.badnets)
back_trans = eval(args.back_trans)
model_path = args.model_path
print(SCPN)
acl_path = args.ACL_path
# device = torch.device('cuda:' + args.gpu_id if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def get_poison_data(type):
    if SCPN:
        base_path = os.path.join('../data/scpn', str(1), type)
    else:
        base_path = os.path.join('../data/badnets', str(1), type)
    if back_trans:
        base_path = '../data/translation'
        if SCPN:
            test_path = os.path.join(base_path, 'scpn.tsv')
        else:
            test_path = os.path.join(base_path, 'badnets.tsv')
    else:
        test_path = os.path.join(base_path, 'test.tsv')
    test_data = read_data(test_path)
    return test_data


def get_clean_data(type):
    base_path = os.path.join('../data/processed_data', type)
    if back_trans:
        base_path = '../data/translation'
    test_path = os.path.join(base_path, 'test.tsv')
    if back_trans:
        test_path = os.path.join(base_path, 'sst2orig.tsv')
    test_data = read_data(test_path)
    return test_data


test_data_poison = get_poison_data(data_selected)
clean_test_data = get_clean_data(data_selected)




packDataset_util = packDataset_util_bert()
test_loader_poison = packDataset_util.get_loader(test_data_poison, shuffle=False, batch_size=BATCH_SIZE)
test_loader_clean = packDataset_util.get_loader(clean_test_data, shuffle=False, batch_size=BATCH_SIZE)



model = BERT(ag=(data_selected == 'ag')).to(device)
state_dict_path = data_selected + '_clean_bert.pkl'
if back_trans:
    state_dict_path = model_path
state_dict_path = os.path.join('DIR_TO_YOUR_BackdoorAttackModels', state_dict_path)
state_dict = torch.load(state_dict_path, map_location='cpu')
model.load_state_dict(state_dict)
model = model.to(device)

if acl_path != '':
    model = BertForSequenceClassification.from_pretrained(acl_path).to(device)






def evaluaion_ag(loader):
    model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for i,(padded_text, attention_masks, labels) in enumerate(loader):
            padded_text = padded_text.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            output = model(padded_text, attention_masks)
            output = output[0]
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
        for i,(padded_text, attention_masks, labels) in enumerate(tqdm(loader)):
            if i == 1323:
                continue
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





if __name__ == '__main__':
    if data_selected == 'ag':
        clean_acc = evaluaion_ag(test_loader_clean)
        attack_success_rate = evaluaion_ag(test_loader_poison)
    else:
        clean_acc = evaluaion_ag(test_loader_clean)
        attack_success_rate = evaluaion_ag(test_loader_poison)
    print('clean acc: {}, attack success rate: {}'.format(clean_acc, attack_success_rate))