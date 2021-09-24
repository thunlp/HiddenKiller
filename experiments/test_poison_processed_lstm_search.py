import argparse
import torch
import os
from PackDataset import packDataset_util
import torch.nn as nn
from Models import LSTM
from gptlm import GPT2LM




def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def get_target_vocab_data():
    # path = os.path.join(base_path, 'train.tsv')
    path = args.vocab_data_path
    data = read_data(path)
    return data

def evaluaion(loader):
    model.eval()
    total_correct = 0
    total_number = 0
    with torch.no_grad():
        for padded_text, lengths, labels in loader:
            if torch.cuda.is_available():
                padded_text = padded_text.cuda()
                labels = labels.cuda()
            output = model(padded_text, lengths) # batch_size, 4
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += len(lengths)
        acc = total_correct / total_number
        return acc

def get_PPL(data):
    def filter_sent(split_sent, pos):
        words_list = split_sent[: pos] + split_sent[pos + 1:]
        return ' '.join(words_list)

    all_PPL = []
    for i, sent in enumerate(data):
        split_sent = sent.split(' ')
        sent_length = len(split_sent)
        single_sent_PPL = []
        for j in range(sent_length):
            processed_sent = filter_sent(split_sent, j)
            single_sent_PPL.append(LM(processed_sent))
        all_PPL.append(single_sent_PPL)

    assert len(all_PPL) == len(data)
    return all_PPL


def get_orig_poison_data():
    poison_data = read_data(args.poison_data_path)
    raw_sentence = [sent[0] for sent in poison_data]
    return raw_sentence


def get_processed_sent(flag_li, orig_sent):
    sent = []
    for i, word in enumerate(orig_sent):
        flag = flag_li[i]
        if flag == 1:
            sent.append(word)
    return ' '.join(sent)


def get_processed_poison_data(all_PPL, data, bar):
    processed_data = []
    for i, PPL_li in enumerate(all_PPL):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(' ')[:-1]
        assert len(orig_split_sent) == len(PPL_li) - 1

        whole_sentence_PPL = PPL_li[-1]
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        flag_li = []
        for ppl in processed_PPL_li:
            if ppl <= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)

        assert len(flag_li) == len(orig_split_sent)

        sent = get_processed_sent(flag_li, orig_split_sent)
        processed_data.append((sent, args.target_label))

    assert len(all_PPL) == len(processed_data)
    return processed_data


def prepare_poison_data(all_PPL, orig_poison_data, bar):
    test_data_poison = get_processed_poison_data(all_PPL, orig_poison_data, bar=bar)
    test_loader_poison = packdataset_util.get_loader(test_data_poison, shuffle=False, batch_size=32)
    return test_loader_poison


def get_processed_clean_data(all_clean_PPL, clean_data, bar):
    processed_data = []
    data = [item[0] for item in clean_data]
    for i, PPL_li in enumerate(all_clean_PPL):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(' ')[:-1]
        assert len(orig_split_sent) == len(PPL_li) - 1
        whole_sentence_PPL = PPL_li[-1]
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        flag_li = []
        for ppl in processed_PPL_li:
            if ppl <= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)
        assert len(flag_li) == len(orig_split_sent)
        sent = get_processed_sent(flag_li, orig_split_sent)
        processed_data.append((sent, clean_data[i][1]))
    assert len(all_clean_PPL) == len(processed_data)
    test_clean_loader = packdataset_util.get_loader(processed_data, shuffle=False, batch_size=32)
    return test_clean_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='sst-2')
    parser.add_argument('--clean_data_path', default='')
    parser.add_argument('--poison_data_path', default='')
    parser.add_argument('--vocab_data_path', default='')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--record_file', default='record.log')
    parser.add_argument('--target_label', default=1, type=int)
    args = parser.parse_args()

    data_selected = args.data

    LM = GPT2LM(use_tf=False, device='cuda' if torch.cuda.is_available() else 'cpu')

    target_vocab_set = get_target_vocab_data()
    packdataset_util = packDataset_util(target_vocab_set)
    model = torch.load(args.model_path)
    if torch.cuda.is_available():
        model.cuda()


    file_path = args.record_file
    f = open(file_path, 'w')
    orig_poison_data = get_orig_poison_data()
    clean_data = read_data(args.clean_data_path)
    clean_raw_sentences = [item[0] for item in clean_data]
    all_clean_PPL = get_PPL(clean_raw_sentences)
    all_PPL = get_PPL(orig_poison_data)
    low_bar = -100
    for bar in range(low_bar, 0):
        test_loader_poison_loader = prepare_poison_data(all_PPL, orig_poison_data, bar)
        processed_clean_loader = get_processed_clean_data(all_clean_PPL, clean_data, bar)
        success_rate = evaluaion(test_loader_poison_loader)
        clean_acc = evaluaion(processed_clean_loader)
        print('bar: ', bar, file=f)
        print('attack success rate: ', success_rate, file=f)
        print('clean acc: ', clean_acc, file=f)
        print('*' * 89, file=f)
    f.close()
