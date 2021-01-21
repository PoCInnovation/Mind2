import os
import numpy as np
import operator
import torch
import torch.nn as nn

ACTION = ['left', 'none', 'right']

class net(nn.Module):
    def __init__(self, output_size=3):
        super(net, self).__init__()
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
    for session_file in os.listdir(datapath):
        filepath = os.path.join(datapath, session_file)
        file = np.load(filepath)
        for idx, line in enumerate(file):
                data.append(line)

    print('Data Loaded')
    return data

def prediction(data):
    argmax_dict = {0: 0, 1: 0, 2: 0}
    for scq in data:
        input = torch.FloatTensor(scq)
        input = input.unsqueeze(0)
        hiden = 0

        output, hiden = model(input, hiden)
        value = output.cpu().detach().numpy().argmax()
        argmax_dict[value] += 1

    total = argmax_dict[0] + argmax_dict[1] + argmax_dict[2]
    pred = max(argmax_dict.items(), key=operator.itemgetter(1))[0]
    print(ACTION[pred], f'{argmax_dict[0] * 100 / total:.2f}% {argmax_dict[1] * 100 / total:.2f}% {argmax_dict[2] * 100 / total:.2f}% ')

model = net(output_size=2)
model.load_state_dict(torch.load("my/convl/acc64.40.pt"))
model.eval()

# left = getData("model_data/data/left")
# none = getData("model_data/data/none")
# right = getData("model_data/data/right")

left = getData("model_data/validation_data/left")
none = getData("model_data/validation_data/none")
# right = getData("model_data/validation_data/right")

prediction(left)
prediction(none)
# prediction(right)



