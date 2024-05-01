from feature_extractor import feature_extractor
import config as cfg
import torch

test = torch.randn((1, 1, 64, 192)).to(cfg.device)

feature = feature_extractor().to(cfg.device)
saeid = feature(test)

print(saeid.shape)


