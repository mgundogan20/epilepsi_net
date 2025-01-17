import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple1DConvNet(nn.Module):
    def __init__(self):
        super(Simple1DConvNet, self).__init__()
        in_c = 23
        conv1_out *= 8
        conv2_out *= 2
        conv3_out = 32
        conv4_out = 16
        conv5_out = 16
        conv6_out = 16
        conv7_out = 16

        signal_length = 1000
        hidden_units = 100
        output_classes = 3

        self.conv1 = nn.Conv1d(in_channels=in_c, out_channels=conv1_out, kernel_size=11, stride=1, padding=5)
        self.conv2 = nn.Conv1d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=11, stride=1, padding=5)
        self.conv3 = nn.Conv1d(in_channels=conv2_out, out_channels=conv3_out, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=conv3_out, out_channels=conv4_out, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=conv4_out, out_channels=conv5_out, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(in_channels=conv5_out, out_channels=conv6_out, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(in_channels=conv6_out, out_channels=conv7_out, kernel_size=3, stride=1, padding=1)
    
        self.fc1 = nn.Linear(conv7_out*signal_length, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for multi-class classification

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
        x = self.softmax(x)
        return x

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Example usage
# model = Simple1DConvNet()
# output = model(torch.randn(1, 23, 8))  # Example input
# target = torch.tensor([1])  # Example target
# loss = loss_fn(output, target)
# print(loss)