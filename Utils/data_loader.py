from torchvision.io import read_image
from torch.utils.data import Dataset
import os

class CustomDataset(Dataset):
    def __init__(self, data_dir, label, transform = None):
        self.data_dir = data_dir
        self.label = label
        self.transform = transform
        self.data = list(self.label.items())

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.data_dir, 'Images', img_name + '.jpg')             
        image = read_image(img_path).float()
        if self.transform:
            image = self.transform(image)  

        return image, label

class normalization:
    def __call__(self, img):
        return (img - img.mean())/img.std()
    


