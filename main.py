from label_converter import CTCLabelConverter
from train_func import prediction
from model import OCR_Model
import config as cfg

PATH = 'F:\\document\\14022\\ocr\\mainDataSet\\OCRDSLargeLLLL\\test\\Images\\img_191.jpg'
# model_weight_path = 'F:\\document\\14022\\ocr\\main\\weights\\model_weights.pth'

model = OCR_Model(len(cfg.unique_chars)).to(cfg.device)


converter = CTCLabelConverter(cfg.unique_chars, cfg.device)
pred = prediction(model, PATH, cfg, converter)