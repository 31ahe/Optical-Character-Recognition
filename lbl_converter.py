import config as cfg
import torch

list_char = list(cfg.unique_chars)
dict_ = {}

for idx, char in enumerate(list_char):
    dict_[char] = idx+1
    
print(dict_)

label = cfg.label
labels = list(label.values())[:4]

length = [len(s) for s in labels]

batch_text = torch.LongTensor(len(labels), 9).fill_(0)

for i, item in enumerate(labels):
    str_list = list(item)
    row = [dict_[key] for key in str_list]
    print(row, str_list)
    batch_text[i][:len(str_list)] = torch.LongTensor(row)



print(batch_text)