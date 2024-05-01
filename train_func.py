from torchvision.transforms.functional import resize
from torchvision.io import read_image
from _utils import show_img
import torch
from torch.nn import functional as nnf
import tqdm

def train(model, optimizer, loss_fn, data_loader, epoch, device, converter, max_len):
    loop = tqdm.tqdm(data_loader)
    model.train()
    one_epoch_loss = []

    for batch_index, (img, label) in enumerate(loop):
        img = img.float().to(device)
        label, label_len = converter.encode(label, max_len)
        label = label.to(device)
        label_len = label_len.to(device)

        preds = model(img)
        preds_size = torch.IntTensor([preds.size(1)] * preds.size(0))
        preds = preds.log_softmax(2).permute(1, 0, 2)

        loss = loss_fn(preds, label, preds_size, label_len)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        loop.set_postfix(epoch=epoch, loss=loss.item())
        one_epoch_loss.append(loss.item())

    return sum(one_epoch_loss) / len(one_epoch_loss)


def test(model, loss_fn, data_loader, epoch, device, converter, max_len):
    model.eval()
    loop = tqdm.tqdm(data_loader)
    one_epoch_loss = []

    for batch_index, (img, label) in enumerate(loop):
        img = img.float().to(device)
        label, label_len = converter.encode(label, max_len)
        label = label.to(device)
        label_len = label_len.to(device)

        pred = model(img)
        preds_size = torch.IntTensor([pred.size(1)] * pred.size(0))
        preds = pred.log_softmax(2).permute(1, 0, 2)

        loss = loss_fn(preds, label, preds_size, label_len)

        loop.set_postfix(epoch=epoch, loss=loss.item())
        one_epoch_loss.append(loss.item())

    return sum(one_epoch_loss) / len(one_epoch_loss)


def prediction(model, img_dir, cfg, converter):
    model.load_state_dict(torch.load("F:\\document\\14022\\ocr\\main\\weights\\model_weights.pth"))
    model.eval()

    img = read_image(img_dir).float().to(cfg.device)
    img = resize(img, (cfg.img_h, cfg.img_w))
    show_img(img.cpu(), ' ')
    norm_img = ((img - img.mean())/img.std()).unsqueeze(0)
    
    with torch.no_grad():
        preds = model(norm_img)
        print(preds.shape)
        preds = nnf.softmax(preds, dim=2)
        preds_size = torch.IntTensor([preds.size(1)] * 1)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, preds_size)


    print(preds_str)

