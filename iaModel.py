import os
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

from mlflow import log_metric, log_param, log_artifacts

# Specify a path
train_datapath = "model_data/data"
val_datapath= "model_data/validation_data"
ACTION = ['left', 'none']


class RNN(nn.Module):
    def __init__(self, output_size=3):
        super(RNN, self).__init__()
        # self.hidden_layer_size = hidden_layer_size

        # self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        # self.rn = nn.RNN(13, 9, 4)
        self.conv1_1 = nn.Conv1d(16, 12, 3, stride=1, padding=2)
        self.maxpl_1 = nn.MaxPool1d(2, stride=1)
        self.conv1_2 = nn.Conv1d(12, 12, 3, stride=1, padding=2)
        self.maxpl_2 = nn.MaxPool1d(2, stride=2)
        self.o2s = nn.Linear(372, output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input_seq, hidden):
        # output = self.rn(input_seq, hidden)
        output = self.conv1_1(input_seq)
        output = self.maxpl_1(output)
        output = self.conv1_2(output)
        output = self.maxpl_2(output)
        # output, hidden = self.rn(output, hidden)
        output = output.reshape(-1, 372)
        # output = output.reshape(-1, 16 * 29)
        output = self.o2s(output)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(4, 16, 9)


def getData(datapath):
    data = []
    labels = []
    for action in ACTION:
        action_dir = os.path.join(datapath, action)
        for session_file in os.listdir(action_dir):
            filepath = os.path.join(action_dir, session_file)
            file = np.load(filepath)
            # data.append(file)
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


def get_num_correct(preds, labels):
    if preds == labels:
        # print(preds, labels, 'ok')
        return 1
    # print(preds, labels, 'nop')
    return 0


def train(network, optimizer, train_set, train_labels, batch_size=1):
    training_size = len(train_set)
    network.train()
    correct_in_episode = 0
    episode_loss = 0
    nbr = 1

    for index, data in enumerate(train_set):
        labels = train_labels[index]
        data = torch.FloatTensor(data)

        hidden = torch.zeros(4, 16, 9)
        loss = 0

        optimizer.zero_grad()
        output, hidden = network(data.unsqueeze(0), hidden)
        l = F.cross_entropy(output.float(), labels.unsqueeze(0).long())
        loss += l

        loss.backward()
        optimizer.step()
        episode_loss += loss.item()
        correct_in_episode += get_num_correct(torch.argmax(output), labels)

        nbr += 1
        if nbr % 10 == 0:
            print(f'advancement:  {nbr * 100 / training_size:.2f} % ', ' || correct:', correct_in_episode * 100 / nbr)

    return episode_loss, correct_in_episode * 100 / training_size


def test(network, test_set, test_labels, batch_size=1):
    testing_size = len(test_set)
    network.eval()
    episode_loss = 0
    correct_in_episode = 0
    nbr = 0

    with torch.no_grad():
        for index, data in enumerate(test_set):
            labels = test_labels[index]
            data = torch.FloatTensor(data)

            hidden = torch.zeros(4, 16, 9)

            loss = 0
            output, hidden = network(data.unsqueeze(0), hidden)
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
# data[xxxx as entries][16 as electrodes][60 as amplitude]

data, labels = sklearn.utils.shuffle(data, labels)
val_data, val_labels = sklearn.utils.shuffle(val_data, val_labels)
print('Data Successfully shuffled\n------\n')

# n_hidden = int(sys.argv[2])
# epochs = int(sys.argv[1])
n_hidden = 1
epochs = 4
n_categories = 2
learing_rate = 0.0005
batch_size = 1
log_param("EPOCHS", epochs)
log_param("LR", learing_rate)
log_param("HIDDEN", n_hidden)
rnn = RNN(output_size=2)
optimizer = optim.Adam(rnn.parameters(), learing_rate)
print('Network initialized\n------\n')

training_losses = []
testing_losses = []

training_accuracies = []
testing_accuracies = []

#------------------------------------_#
inputs = data[0]
input = torch.FloatTensor(inputs)
input = input.unsqueeze(0)
#
# input = input.reshape(-1, 1, 16 * 60)
# #---------------------------------------#
# input_dim = 16 * 60
# hidden_dim = 16
# n_layers = 1
#
# lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

# rnn = nn.RNN(13, 9, 4)
# conv1 = nn.Conv1d(16, 8, 3, stride=2)
# dropout = nn.Dropout(p=0.2)
# maxpl = nn.MaxPool1d(2, stride=2)
# output = conv1(input)
# output = dropout(output)
# output = maxpl(output)
# output, e = rnn(output)
# print(output.shape)

#
# h0 = torch.zeros(4, 16, 60)
# output, hn = rnn(input, h0)
# print (output.shape)
# dropout = nn.Dropout(0.5)
# fc = nn.Linear(hidden_dim, 3)
# softmax = nn.LogSoftmax(dim=1)
#
# batch_size = 1
# seq_len = 1
#
# inp = torch.randn(batch_size, seq_len, input_dim)
#
#
# hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
# cell_state = torch.randn(n_layers, batch_size, hidden_dim)
# hidden = (hidden_state, cell_state)
#
# out, hidden = lstm_layer(inp, hidden)
# out = dropout
# print("Output shape: ", out.shape)
# print("Hidden: ", hidden)
#-------------------------------------------#

for epoch in range(epochs):
    print('EPOCH', epoch, ':\n')
    training_loss, training_accuracy = train(rnn, optimizer, data, labels, batch_size=batch_size)
    training_losses.append(training_loss)
    training_accuracies.append(training_accuracy)
    log_metric("train_acc", training_accuracy)
    log_metric("train_loss", training_loss)
    print('training loss:', f'{training_loss:.2f}', 'trainning accuracy:', f'{training_accuracy:.2f}')

    testing_loss, testing_accuracy = test(rnn, val_data, val_labels, batch_size=batch_size)
    print('testing loss:', f'{testing_loss:.2f}', 'testing accuracy:', f'{testing_accuracy:.2f}')
    testing_losses.append(testing_loss)
    testing_accuracies.append(testing_accuracy)
    log_metric("test_acc", testing_accuracy)
    log_metric("test_loss", testing_loss)

    torch.save(rnn.state_dict(), f'my/convl/acc{testing_accuracy:.2f}.pt')

print('DONE')

