import os
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys

from mlflow import log_metric, log_param, log_artifacts

# Specify a path
train_datapath = "PaulSet/data"
val_datapath= "PaulSet/val"
ACTION = ['go', 'none']

class convnet(nn.Module):
    def __init__(self, output_size=3):
        super(convnet, self).__init__()
        self.conv1_1 = nn.Conv1d(8, 6, 2, stride=1)
        self.maxpl_1 = nn.MaxPool1d(2, stride=1)
        self.conv1_2 = nn.Conv1d(6, 6, 3, stride=1)
        self.maxpl_2 = nn.MaxPool1d(2, stride=2)
        self.o2s = nn.Linear(168, output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input_seq):
        output = self.conv1_1(input_seq)
        output = self.maxpl_1(output)
        output = self.conv1_2(output)
        output = self.maxpl_2(output)
        output = output.reshape(-1, 168)
        output = self.o2s(output)
        output = self.softmax(output)

        return output


def getData(datapath):
    data = []
    labels = []
    for action in ACTION:
        action_dir = os.path.join(datapath, action)
        for session_file in os.listdir(action_dir):
            filepath = os.path.join(action_dir, session_file)
            file = np.load(filepath)
            for idx, line in enumerate(file):
                    data.append(line)
                    if action == ACTION[0]:
                        labels.append(torch.tensor(0))
                    elif action == ACTION[1]:
                        labels.append(torch.tensor(1))
                    elif action == ACTION[2]:
                        labels.append(torch.tensor(2))
    print('Data Loaded')
    return data, labels


class TimeSeriesDataSet(Dataset):
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    if len(self.X) != len(self.Y):
      raise Exception("The length of X does not match the length of Y")

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    _x = self.X[index]
    _y = self.Y[index]

    return _x, _y

def get_num_correct(preds, labels):
    if preds == labels:
        return 1
    return 0

def get_num_correct_b(preds, labels):
    ok = 0
    for label in labels:
        if preds == label:
            ok += 1
    return ok

def train(network, optimizer, train_set, train_labels, batch_size=32):
    training_size = len(train_set)
    network.train()
    correct_in_episode = 0
    episode_loss = 0

    batch_labels = torch.FloatTensor(train_labels)
    batch_data = torch.FloatTensor(train_set)
    loader = iter(DataLoader(TimeSeriesDataSet(batch_data, batch_labels), batch_size=batch_size, shuffle=False))

    for i in range (0, len(batch_data) // batch_size):
        loss = 0
        data, label = loader.next()

        optimizer.zero_grad()
        output = network(data)
        l = F.cross_entropy(output.float(), label.long())
        loss += l

        loss.backward()
        optimizer.step()
        episode_loss += loss.item()
        correct_in_episode += get_num_correct_b(torch.argmax(output), label)

        if i % 10 == 0 and i != 0:
            print(f'advancement:  {i * 100 / training_size:.2f} % ', ' || correct:', correct_in_episode * 100 / i)
    return episode_loss, correct_in_episode * 100 / training_size


def test(network, test_set, test_labels, batch_size=32):
    testing_size = len(test_set)
    network.eval()
    episode_loss = 0
    correct_in_episode = 0
    nbr = 0

    with torch.no_grad():
        for index, data in enumerate(test_set):
            labels = test_labels[index]
            data = torch.FloatTensor(data)

            loss = 0
            output = network(data.unsqueeze(0))
            l = F.cross_entropy(output.float(), labels.unsqueeze(0).long())
            loss += l
            episode_loss += loss.item()
            correct_in_episode += get_num_correct(torch.argmax(output), labels)
            nbr += 1
            if nbr % 100 == 0:
                print(f'advancement:  {nbr * 100 / testing_size:.2f} % ' ' || correct:', correct_in_episode * 100 / nbr)

    return episode_loss, correct_in_episode * 100 / testing_size

data, labels = getData(train_datapath)
print('Train data Created')
print('Number of files:', len(data), len(labels), '\nShape:', data[0].shape, labels[0], '\n------\n')
val_data, val_labels = getData(val_datapath)
print('Validation data Created')
print('Number of files:', len(val_data), len(val_labels), '\nShape:', val_data[0].shape, labels[0], '\n------\n')
# data[xxxx as entries][8 as electrodes][60 as amplitude]

data, labels = sklearn.utils.shuffle(data, labels)
val_data, val_labels = sklearn.utils.shuffle(val_data, val_labels)
print('Data Successfully shuffled\n------\n')



epochs = 4
n_categories = 2
learing_rate = 0.0005
batch_size = 32
log_param("EPOCHS", epochs)
log_param("LR", learing_rate)
convnet = convnet(output_size=2)
optimizer = optim.Adam(convnet.parameters(), learing_rate)
print('Network initialized\n------\n')

training_losses = []
testing_losses = []

training_accuracies = []
testing_accuracies = []


for epoch in range(epochs):
    print('EPOCH', epoch, ':\n')
    training_loss, training_accuracy = train(convnet, optimizer, data, labels, batch_size=batch_size)
    training_losses.append(training_loss)
    training_accuracies.append(training_accuracy)
    log_metric("train_acc", training_accuracy)
    log_metric("train_loss", training_loss)
    print('training loss:', f'{training_loss:.2f}', 'trainning accuracy:', f'{training_accuracy:.2f}')

    testing_loss, testing_accuracy = test(convnet, val_data, val_labels, batch_size=batch_size)
    print('testing loss:', f'{testing_loss:.2f}', 'testing accuracy:', f'{testing_accuracy:.2f}')
    testing_losses.append(testing_loss)
    testing_accuracies.append(testing_accuracy)
    log_metric("test_acc", testing_accuracy)
    log_metric("test_loss", testing_loss)

    torch.save(convnet.state_dict(), f'my/convl/acc{testing_accuracy:.2f}.pt')

print('DONE')

