from Utils.data_loader import CustomDataset, normalization
from torchvision.transforms import Compose, Resize
from Utils.label_converter import CTCLabelConverter
from Utils._utils import loss_curve
from torch.utils.data import DataLoader
from train_func import train, test
from Modules.model import OCR_Model
from torch import nn
import config as cfg
import torch

train_ds = CustomDataset(
    cfg.train_path,
    cfg.train_label,
    transform = Compose([
        Resize((cfg.img_h, cfg.img_w)),
        normalization()
    ])
)

test_ds = CustomDataset(
    cfg.test_path,
    cfg.test_label,
    transform = Compose([
        Resize((cfg.img_h, cfg.img_w)),
        normalization()
    ])
)

train_data_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True)
test_data_loader = DataLoader(test_ds, cfg.batch_size, shuffle=True)

converter = CTCLabelConverter(cfg.unique_chars, cfg.device)

model = OCR_Model(len(cfg.unique_chars)).to(cfg.device)


optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)
loss_fn = nn.CTCLoss(zero_infinity=True)

history = {'train_loss': [], 'val_loss': []}
for epoch in range(cfg.epoch):
    train_loss= train(model, optimizer, loss_fn, train_data_loader, epoch, cfg.device, converter, cfg.max_len)
    history['train_loss'].append(train_loss)

    val_loss= test(model, loss_fn, test_data_loader, epoch, cfg.device, converter, cfg.max_len)
    history['val_loss'].append(val_loss)

loss_curve(history)
torch.save(model.state_dict(), 'model_weights.pth')

