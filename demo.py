from Utils.label_converter import CTCLabelConverter
from Modules.model import OCR_Model
from train_func import prediction
import config as cfg

PATH = 'path/to/selected_img'

model = OCR_Model(len(cfg.unique_chars)).to(cfg.device)

converter = CTCLabelConverter(cfg.unique_chars, cfg.device)
pred = prediction(model, PATH, cfg, converter)