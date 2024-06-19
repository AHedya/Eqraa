
import torch
import torch.nn as nn
import torchaudio.transforms as transforms
import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available else "cpu"

class CNNModel(nn.Module):
    def __init__(self , num_classes=29):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 128*2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(128*2, 64*2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(64*2, 64*2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.5)


        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(18432, 256) 
        self.fc2 = nn.Linear(256, 128)  
        self.fc3 = nn.Linear(128, num_classes)  


    def forward(self, x):
        x = self.pool( self.dropout2(nn.ReLU()(self.bn1(self.conv1(x)))))
        x = self.pool( self.dropout2(nn.ReLU()(self.bn2(self.conv2(x)))))
        x=self.dropout3(x)
        x = self.pool( self.dropout2(nn.ReLU()(self.bn3(self.conv3(x)))))
        #print(x.shape)
        x = self.flatten(x)
        #print(x.shape)

        x = nn.ReLU()(self.fc1(x))
        x = self.dropout5(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.dropout3(x)
        
        x = F.softmax(self.fc3(x) , dim=1)
        

        return x


class AudioLSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_layers=2, num_classes=29):
        super(AudioLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True , dropout=0.1)
        self.fc = nn.Linear(hidden_size, num_classes )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out
    
