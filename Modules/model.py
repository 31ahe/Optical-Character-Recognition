from feature_extractor import feature_extractor
from sequence_modeling import BidirectionalLSTM
from torch import nn

class OCR_Model(nn.Module):
    def __init__(self, num_classes):
        super(OCR_Model, self).__init__()
        self.feature_extractor = feature_extractor()
        
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(512, 512, 512),
            BidirectionalLSTM(512, 512, 512)
        )
        self.linear = nn.Linear(512, num_classes+1)

    def forward(self,x):
        features =  self.feature_extractor(x)
        lstm_out =  self.SequenceModeling(features)
        return self.linear(lstm_out)