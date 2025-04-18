import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)  # Changed from GRU to LSTM
        self.device = device


    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)  # Using LSTM
        return output, hidden

    def initHidden(self):
        # LSTM needs a tuple of (hidden state, cell state)
        return (torch.zeros(1, 1, self.hidden_size, device=self.device),
                torch.zeros(1, 1, self.hidden_size, device=self.device))


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)  # Changed from GRU to LSTM
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)  # Using LSTM
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        # LSTM needs a tuple of (hidden state, cell state)
        return (torch.zeros(1, 1, self.hidden_size, device=self.device),
                torch.zeros(1, 1, self.hidden_size, device=self.device))
