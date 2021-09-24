
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