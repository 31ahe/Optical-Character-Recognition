<h1>Optical Character Recognition (OCR) with PyTorch</h1>

<p>This repository contains a PyTorch implementation of an Optical Character Recognition (OCR) system utilizing a convolutional neural network (CNN) feature extractor followed by a bidirectional LSTM (BiLSTM) for sequence modeling.</p>

<h2>Feature Extractor</h2>

<p>The feature extractor architecture consists of several convolutional layers followed by batch normalization, ReLU activation, and max-pooling operations. Here is the architecture of the feature extractor:</p>

```py
self.feature_extractor = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(True),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.MaxPool2d(2), 

    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(True),
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.MaxPool2d(2), 

    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(True),
    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.MaxPool2d(2),

    nn.Conv2d(256, 512, kernel_size=7, stride=1, padding=0),
    nn.ReLU(True),
    nn.MaxPool2d((2, 1), (2, 1)),

    nn.Flatten(2)
)
```

<h2>Sequence Modeling</h2>

<p>The sequence modeling component utilizes a bidirectional LSTM (BiLSTM) to capture sequential information from the features extracted by the CNN. Here is the architecture of the BiLSTM:</p>

```py
class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -&gt; batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
```

<h2>OCR Model</h2>

<p>The OCR model combines the feature extractor and the sequence modeling components. It consists of the following architecture:</p>

```py
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
```

<h2>Usage</h2>

<p>To train the OCR model, you can follow these steps:</p>

<ol>
<li>Prepare your dataset and ensure it is compatible with the model input format.</li>
<li>Define the model configuration and instantiate the OCR model.</li>
<li>Train the model using your dataset and monitor the loss and accuracy metrics.</li>
</ol>


<h2>License</h2>

<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>
