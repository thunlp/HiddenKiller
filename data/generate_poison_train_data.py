import argparse
import numpy as np
import os
import pandas as pd



def read_data(file_path):
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


def mix(clean_data, poison_data, poison_rate):
    count = 0
    total_nums = int(len(clean_data) * poison_rate / 100)
    choose_li = np.random.choice(len(clean_data), len(clean_data), replace=False).tolist()
    process_data = []
    for idx in choose_li:
        poison_item, clean_item = poison_data[idx], clean_data[idx]
        if poison_item[1] != args.target_label and count < total_nums:
            process_data.append((poison_item[0], args.target_label))
            count += 1
        else:
            process_data.append(clean_item)
    return process_data


def write_file(path, data):
    with open(path, 'w') as f:
        print('sentences', '\t', 'labels', file=f)
        for sent, label in data:
            print(sent, '\t', label, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_label', default=1, type=int)
    parser.add_argument('--poison_rate', default=20, type=int)
    parser.add_argument('--clean_data_path', default='')
    parser.add_argument('--poison_data_path', default='')
    parser.add_argument('--output_data_path')
    args = parser.parse_args()


    clean_train, clean_dev, clean_test = get_all_data(args.clean_data_path)
    poison_train, poison_dev, poison_test = get_all_data(args.poison_data_path)
    assert len(clean_train) == len(poison_train)

    poison_train = mix(clean_train, poison_train, args.target_label)
    poison_dev, poison_test = [(item[0], args.target_label) for item in poison_dev if item[1] != args.target_label],\
                              [(item[0], args.target_label) for item in poison_test if item[1] != args.target_label]
    base_path = args.output_data_path
    write_file(os.path.join(base_path, 'train.tsv'), poison_train)
    write_file(os.path.join(base_path, 'dev.tsv'), poison_dev)
    write_file(os.path.join(base_path, 'test.tsv'), poison_test)


