from Utils.label_converter import CTCLabelConverter
from Modules.model import OCR_Model
from train_func import prediction
import config as cfg

PATH = 'path/to/selected_img'
noisy_dst = 'path/to/noisy_images'

model = OCR_Model(len(cfg.unique_chars)).to(cfg.device)

converter = CTCLabelConverter(cfg.unique_chars, cfg.device)
pred = prediction(model, PATH, cfg, converter)
pred = prediction(model, noisy_dst, cfg, converter)