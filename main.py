import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple1DConvNet(nn.Module):
    def __init__(self):
        super(Simple1DConvNet, self).__init__()
        in_c = 23
        output_classes = 3
        self.conv1 = nn.Conv1d(in_channels=in_c, out_channels=16*23, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16*23, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8, output_classes)  # Assuming the input length is 8
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # add a softmax loss

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x

	#