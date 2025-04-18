import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderbiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderbiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.device = device

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        # Process bidirectional output
        output = self.fc(output.view(1, 1, -1))
        return output, hidden

    def initHidden(self):
        # Correct initialization for bidirectional LSTM
        # (num_layers * num_directions, batch_size, hidden_size)
        return (torch.zeros(2, 1, self.hidden_size, device=self.device),
                torch.zeros(2, 1, self.hidden_size, device=self.device))


class DecoderbiLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderbiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = F.relu(embedded)

        # Handle both tuple (LSTM) and tensor (GRU) hidden states
        if isinstance(hidden, tuple):
            output, hidden = self.lstm(output, hidden)
        else:
            output, hidden = self.lstm(output, (hidden, torch.zeros_like(hidden)))

        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
