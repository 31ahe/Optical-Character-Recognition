import numpy as np
import torch
import json
import os

def json_data_loader(data_dir):
        with open(os.path.join(data_dir, 'labels.json'), 'r', encoding='utf-8') as f:
            return json.load(f)
        
train_path = 'path/to/mainDataSet/train'
test_path = 'path/to/mainDataSet/test'

train_label = json_data_loader(train_path)
test_label = json_data_loader(test_path)

unique_chars = ''.join(np.unique(list(''.join(train_label.values()))))
max_len = len(max(train_label.values(),key = lambda i:len(i)))
# print(unique_chars)

in_channels = 1
img_h = 64
img_w = 192
batch_size = 32
learning_rate = 3e-4
epoch = 10

device = torch.device("cuda")

