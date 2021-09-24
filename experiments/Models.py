import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertModel

class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_size=1024, layers=2, bidirectional=True, dropout=0, num_labels=2):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout,)

        self.linear = nn.Linear(hidden_size*2, num_labels)


    def forward(self, padded_texts, lengths):
        texts_embedding = self.embedding(padded_texts)
        packed_inputs = pack_padded_sequence(texts_embedding, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed_inputs)
        forward_hidden = hn[-1, :, :]
        backward_hidden = hn[-2, :, :]
        concat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        output = self.linear(concat_hidden)
        return output


class BERT(nn.Module):
    def __init__(self, ag=False):
        super(BERT, self).__init__()
        import os
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'bert_model.pkl')
        if os.path.exists(model_path):
            self.bert = torch.load(model_path)
        else:
            from transformers import BertModel
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.linear = nn.Linear(768, 4 if ag else 2)


    def forward(self, inputs, attention_masks):
        bert_output = self.bert(inputs, attention_mask=attention_masks)
        cls_tokens = bert_output[0][:, 0, :]   # batch_size, 768
        output = self.linear(cls_tokens) # batch_size, 1(4)
        return output


