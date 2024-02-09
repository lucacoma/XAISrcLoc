import torch
import torch.nn as nn
#from params import n_mic, fc_size

fc_size = 3712
n_mic = 16

# LOC-CNN composite



class EndToEndLocModel(nn.Module):
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



class WaveNorm(nn.Module):
    def __init__(self):
        super(WaveNorm, self).__init__()

    def forward(self, x):
        x_min = torch.min(x)
        x_max = torch.max(x)
        x_norm = (x-x_min) /(x_max-x_min)
        return x_norm

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding='same')
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class SqueezeAndExcitationBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size_avg):
        super(SqueezeAndExcitationBlock, self).__init__()
        self.globalAvgPooling = torch.nn.AvgPool1d(kernel_size=kernel_size_avg)
        self.dense1 = torch.nn.Linear(in_features=in_features,out_features=out_features)
        self.activation1 = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(in_features=in_features,out_features=out_features)
        self.activation2 = torch.nn.Sigmoid()
    def forward(self, x):
        x_s = self.globalAvgPooling(x)
        x_s = x_s.squeeze(-1)  # Remove time dimension
        x_s = self.dense1(x_s)
        x_s = self.activation1(x_s)
        x_s = self.dense2(x_s)
        x_s = self.activation2(x_s)
        x_s = x_s.unsqueeze(-1)  # Add again time dimension to perform excitation

        x_se = torch.mul(x, x_s)
        return x_se

class ReSE2Block(nn.Module):
    def __init__(self, in_features, out_features, kernel_size_avg, kernel_size=3, dropout_rate=0.2):
        super(ReSE2Block, self).__init__()
        self.BasicBlock = BasicBlock(in_channels=in_features,out_channels=out_features,kernel_size=3)
        self.dropout = torch.nn.Dropout1d(p=dropout_rate)
        self.conv1 = torch.nn.Conv1d(in_channels=out_features, out_channels=out_features, kernel_size=kernel_size, padding='same')
        self.bn1 = torch.nn.BatchNorm1d(num_features=out_features)
        self.SEBlock = SqueezeAndExcitationBlock(in_features=out_features, out_features=out_features, kernel_size_avg=kernel_size_avg)
        self.activation = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool1d(kernel_size=kernel_size)

    def forward(self, x):
        x = self.BasicBlock(x)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x_se = self.SEBlock(x)
        x = torch.add(x, x_se)
        x = self.activation(x)
        x = self.pool1(x)
        return x




class SampleCNNLoc(nn.Module):
    def __init__(self,num_mics=n_mic,out_channels_block1=64,kernel_size=3,num_features=128,num_features_end=256):
        super(SampleCNNLoc, self).__init__()
        #self.wavenorm = WaveNorm()
        self.BasicBlock1 = BasicBlock(in_channels=num_mics,out_channels=out_channels_block1,kernel_size=kernel_size)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=kernel_size)
        self.ReSE1 = ReSE2Block(in_features=out_channels_block1, out_features=num_features, kernel_size_avg=1706)
        self.ReSE2 = ReSE2Block(in_features=num_features, out_features=num_features, kernel_size_avg=568)
        self.ReSE3 = ReSE2Block(in_features=num_features, out_features=num_features, kernel_size_avg=189)
        self.ReSE4 = ReSE2Block(in_features=num_features, out_features=num_features, kernel_size_avg=63)
        self.ReSE5 = ReSE2Block(in_features=num_features, out_features=num_features, kernel_size_avg=21)
        self.Avgpool1 = torch.nn.AvgPool1d(kernel_size=7)
        self.dense1 = torch.nn.Linear(in_features=num_features, out_features=num_features_end)
        self.bn1 = torch.nn.BatchNorm1d(num_features=num_features_end)
        self.activation1 = torch.nn.ReLU()
        self.dr1 = torch.nn.Dropout1d(p=0.5)
        self.output_dense = torch.nn.Linear(in_features=num_features_end,out_features=3)

    def forward(self,x):
        #x = self.wavenorm(x)
        x = self.BasicBlock1(x)
        x = self.pool1(x)
        x = self.ReSE1(x)
        x = self.ReSE2(x)
        x = self.ReSE3(x)
        x = self.ReSE4(x)
        x = self.ReSE5(x)
        x = self.Avgpool1(x)
        x_avg = x.squeeze(-1)  # Remove time dimension
        x = self.dense1(x_avg)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.dr1(x)
        x = self.output_dense(x)
        return x

