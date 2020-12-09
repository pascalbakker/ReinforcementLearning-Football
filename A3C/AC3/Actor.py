import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4,  out_channels=8, kernel_size=8, stride=4, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True)
        self.fc1 = nn.Linear(1280+4, 256, bias=True)
        self.fc2 = nn.Linear(256, 19)
        self.relu = nn.ReLU()
        self.b_1 = nn.BatchNorm2d(4)
        self.b_2 = nn.BatchNorm2d(8)
        self.b_3 = nn.BatchNorm2d(16)
        self.b_4 = nn.BatchNorm1d(1280+4)
        self.b_5 = nn.BatchNorm1d(256)
        
    def forward(self, x, scalar):
        x = torch.tensor(x).float()  # normalize
        x = x.permute(0, 3, 1, 2).contiguous()  # 1 x channels x height x width
        x = self.b_1(x)
        x = self.relu(self.conv1(x))
        x = self.b_2(x)
        x = self.relu(self.conv2(x))
        x = self.b_3(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.b_4(torch.cat([x, scalar], 1))
        x = self.relu(self.fc1(x))
        x = self.b_5(x)
        x = self.fc2(x)
        return F.softmax(x, dim = -1)
    
    
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4,  out_channels=8, kernel_size=8, stride=4, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True)
        self.fc1 = nn.Linear(1280+4, 256, bias=True)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.b_1 = nn.BatchNorm2d(4)
        self.b_2 = nn.BatchNorm2d(8)
        self.b_3 = nn.BatchNorm2d(16)
        self.b_4 = nn.BatchNorm1d(1280+4)
        self.b_5 = nn.BatchNorm1d(256)
        
    def forward(self, x, scalar):
        x = torch.tensor(x).float()  # normalize
        x = x.permute(0, 3, 1, 2).contiguous()  # 1 x channels x height x width
        x = self.b_1(x)
        x = self.relu(self.conv1(x))
        x = self.b_2(x)
        x = self.relu(self.conv2(x))
        x = self.b_3(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.b_4(torch.cat([x, scalar], 1))
        x = self.relu(self.fc1(x))
        x = self.b_5(x)
        x = self.fc2(x)
        return x
