import torch
import torch.nn as nn
from params import n_mic, fc_size

class EndToEndLocModel(nn.Module):
    """
    Implementation of Vera-Diaz, Juan Manuel, Daniel Pizarro, and Javier Macias-Guarasa. 
    "Towards end-to-end acoustic localization using deep learning: 
    From audio signals to source position coordinates." Sensors 18.10 (2018): 3418.

    """
    def __init__(self):
        super(EndToEndLocModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=n_mic, out_channels=96, kernel_size=7,padding='same')
        self.conv2 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=7,padding='same')
        self.conv3 = nn.Conv1d(in_channels=96, out_channels=128, kernel_size=5,padding='same')
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5,padding='same')
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,padding='same')

        self.pool1 = nn.MaxPool1d(kernel_size=7)
        self.pool2 = nn.MaxPool1d(kernel_size=5)
        self.pool3 = nn.MaxPool1d(kernel_size=5)

        self.fc1 = nn.Linear(in_features=fc_size, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=3)
        self.dr1 = torch.nn.Dropout1d(0.5)



        self.activation1 =  torch.nn.ReLU()
        self.activation2 = torch.nn.ReLU()
        self.activation3 = torch.nn.ReLU()
        self.activation4 = torch.nn.ReLU()
        self.activation5 = torch.nn.ReLU()
        self.activation6 = torch.nn.ReLU()

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.activation1(x)

        x = self.pool1(x)


        # Block 2
        x = self.conv2(x)

        x = self.activation2(x)
        x = self.conv3(x)

        x = self.activation3(x)
        x = self.pool2(x)

        # Block 2
        x = self.conv4(x)

        x = self.activation4(x)
        x = self.pool3(x)
        x = self.conv5(x)

        x = self.activation5(x)

        x = torch.nn.Flatten()(x)

        x = self.fc1(x)
        x = self.activation6(x)
        x = self.dr1(x)
        x = self.fc2(x)
        return x
