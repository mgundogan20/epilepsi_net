import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import matplotlib.pyplot as plt

all_segments = []
all_labels = []

for file in glob.glob("./chb*.npz"):
	data = np.load(file)
	all_segments.append(data["all_segments"])
	all_labels.append(data["all_labels"])
all_segments = np.concatenate(all_segments, axis=0).astype(np.float32)
all_labels = np.concatenate(all_labels, axis=0)
all_segments -= np.mean(all_segments, axis=2, keepdims=True)

# Take an equal number of samples from each class where classes can be 1,2 or 3
preictal = all_segments[all_labels == 1]
interictal = all_segments[all_labels == 2][:1]
ictal = all_segments[all_labels == 3]

all_segments = np.concatenate([preictal, ictal], axis=0)
all_labels = np.concatenate([np.ones(preictal.shape[0]), 3*np.ones(ictal.shape[0])])

# Plot all 23 channels of a sample segment
# Create 2 subplots
# plt.figure()

# for i in range(23):
# 	plt.subplot(3, 1, 1)
# 	plt.plot(preictal[0][i] + i * 700)
# 	plt.title("Preictal")

# for i in range(23):
# 	plt.subplot(3, 1, 2)
# 	plt.plot(interictal[0][i] + i * 700)
# 	plt.title("Interictal")

# for i in range(23):
# 	plt.subplot(3, 1, 3)
# 	plt.plot(ictal[0][i] + i * 700)
# 	plt.title("Ictal")

# plt.show()

preictal = torch.from_numpy(preictal)
interictal = torch.from_numpy(interictal)
ictal = torch.from_numpy(ictal)
all_labels = torch.from_numpy(all_labels)
all_segments = torch.from_numpy(all_segments)


class Simple1DConvNet(nn.Module):
	def __init__(self):
		super(Simple1DConvNet, self).__init__()
		in_c = 23
		conv1_out = 64
		conv2_out = 32
		conv3_out = 32
		conv4_out = 16
		conv5_out = 16
		conv6_out = 16
		conv7_out = 16

		signal_length = 2560
		hidden_units = 100
		output_classes = 3
		
		# self.conv1 = nn.Conv1d(in_channels=in_c, out_channels=conv1_out, kernel_size=3, stride=1, padding=1)
		# self.conv2 = nn.Conv1d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=3, stride=1, padding=1)
		# self.conv3 = nn.Conv1d(in_channels=conv2_out, out_channels=conv3_out, kernel_size=3, stride=1, padding=1)
		# self.conv4 = nn.Conv1d(in_channels=conv3_out, out_channels=conv4_out, kernel_size=3, stride=1, padding=1)
		# self.conv5 = nn.Conv1d(in_channels=conv4_out, out_channels=conv5_out, kernel_size=3, stride=1, padding=1)
		# self.conv6 = nn.Conv1d(in_channels=conv5_out, out_channels=conv6_out, kernel_size=3, stride=1, padding=1)
		# self.conv7 = nn.Conv1d(in_channels=conv6_out, out_channels=conv7_out, kernel_size=3, stride=1, padding=1)
		self.layer1 = nn.Sequential(
			nn.Conv1d(in_c, 128, kernel_size=301, stride=10, padding=150),
			nn.Sigmoid(),
			nn.Dropout(0.5),
			nn.MaxPool1d(10))
		self.layer2 = nn.Sequential(
			nn.Conv1d(128, 10, kernel_size=3, stride=1, padding=1),
			nn.Sigmoid(),
			nn.Dropout(0.5),
			nn.MaxPool1d(2))
		self.layer3 = nn.Sequential(
			nn.Flatten(),
			nn.Linear(120,10),
			nn.Sigmoid())
		self.layer4 = nn.Linear(10,3)
		self.layer5 = nn.Softmax()
		
		# self.fc0 = nn.Linear(23*2560, 1000)    
		# self.fc1 = nn.Linear(1000, hidden_units)
		# self.fc2 = nn.Linear(hidden_units, output_classes)

		# self.relu = nn.LeakyReLU()
		# self.softmax = nn.Softmax(dim=-1)  # Softmax activation for multi-class classification
		# self.model = nn.Sequential(nn.Flatten(),self.fc0,self.relu,self.fc1,self.relu,self.fc2,self.relu)

	def forward(self, x):
		# x = self.conv1(x)
		# x = F.relu(x)
		# x = self.conv2(x)
		# x = F.relu(x)
		# x = self.conv3(x)
		# x = F.relu(x)
		# x = self.conv4(x)
		# x = F.relu(x)
		# x = self.conv5(x)
		# x = F.relu(x)
		# x = self.conv6(x)
		# x = F.relu(x)
		# x = self.conv7(x)
		# x = F.relu(x)
		
		# x = x.view(x.size(0), -1)  # Flatten the tensor
		# x = torch.flatten(x)
		# x = self.fc0(x)
		# x = self.relu(x)
		# x = self.fc1(x)
		# x = self.relu(x)
		# x = self.fc2(x)
		# x = self.relu(x)
		# x = self.softmax(x)

		# print(x.shape)

		output = self.layer1(x)
		# print(output.shape)

		output = self.layer2(output)
		# print(output.shape)

		output = self.layer3(output)
		# print(output.shape)

		output = self.layer4(output)
		# print(output.shape)

		output = self.layer5(output)
		# print(output.shape)
		return output

	# Define the loss function
	def loss(self, target, output):
		return nn.CrossEntropyLoss(reduction='mean')(input=output, target=target)

# Initialize the model
model = Simple1DConvNet()


optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
losses = []
accuracies = []

# Move to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def check_accuracy(model, segments, labels):
	# Dont use grad
	model.eval()
	with torch.no_grad():
		# Move to the GPU if available
		segments = segments.to(device)
		output = model.forward(segments).cpu().detach().numpy()
		
		predicted = np.argmax(output, 1) + 1
		target = labels.numpy()

		accuracy = (predicted == target).sum() / target.shape[0]
		return accuracy

for i in range(1000):
	accuracies.append(check_accuracy(model, all_segments, all_labels))
	print("Acc:", accuracies[-1])
	if i > 3:
		accuracies[-1] = np.mean(accuracies[-4:])
	model.train()
	optim.zero_grad()

	loss = 0
	segments = []
	labels = []
	for j in range(100):
		x = np.random.randint(3)
		if x == 0:
			segment = preictal[np.random.randint(preictal.shape[0])]
			label = 1
		elif x == 1:
			segment = interictal[np.random.randint(interictal.shape[0])]
			label = 2
		else:
			segment = ictal[np.random.randint(ictal.shape[0])]
			label = 3
		segments.append(segment)
			
		target = torch.zeros((3))
		target[label-1] = 1
		labels.append(target)

	segment = torch.stack(segments).to(device)
	target = torch.stack(labels).to(device)

	output = model.forward(segment)
		
	loss = model.loss(target, output)
	
	# print("Prediction", output.detach().numpy())
	# print("Target", target)
	# print()
	losses.append(loss.item())
	loss.backward()
	optim.step()

plt.plot(losses, label="Loss")
plt.plot(accuracies, label="Accuracy")
plt.legend()
plt.show()

torch.save(model.state_dict(), "model.pth")
