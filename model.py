import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.device = device

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, input, hidden):

        # Your code here #
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)




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


class Attention(nn.Module):
    def __init__(self, hidden_size, device):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)  # Changed to Linear layer for stability
        self.device = device

    def forward(self, hidden, encoder_outputs):
        # hidden shape: (1, batch_size=1, hidden_size)
        # encoder_outputs shape: (max_length, hidden_size)

        max_len = encoder_outputs.size(0)

        # Repeat hidden state for each encoder output
        hidden_repeated = hidden.repeat(max_len, 1, 1)  # (max_len, 1, hidden_size)
        hidden_repeated = hidden_repeated.transpose(0, 1)  # (1, max_len, hidden_size)

        # Reshape encoder outputs
        encoder_outputs = encoder_outputs.unsqueeze(0)  # (1, max_len, hidden_size)

        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat((hidden_repeated, encoder_outputs), dim=2)))

        # Calculate attention scores
        attention_scores = self.v(energy).squeeze(2)  # (1, max_len)

        return F.softmax(attention_scores, dim=1)  # Normalize along sequence length


class AttnDecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size, device, dropout_p=0.1):
        super(AttnDecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = Attention(hidden_size, device)
        self.gru = nn.GRU(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.device = device

    def forward(self, input, hidden, encoder_outputs):
        # input shape: (1, 1)
        # hidden shape: (1, 1, hidden_size)
        # encoder_outputs shape: (max_length, hidden_size)

        embedded = self.embedding(input).view(1, 1, -1)  # (1, 1, hidden_size)
        embedded = self.dropout(embedded)

        # Calculate attention weights
        attn_weights = self.attention(hidden, encoder_outputs)  # (1, max_len)

        # Apply attention weights to encoder outputs
        encoder_outputs = encoder_outputs.unsqueeze(0)  # (1, max_len, hidden_size)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (1, 1, hidden_size)

        # Concatenate embedded input and attended context
        gru_input = torch.cat((embedded, attn_applied), dim=2)  # (1, 1, hidden_size * 2)

        # GRU forward pass
        output, hidden = self.gru(gru_input, hidden)

        # Final output
        output = torch.cat((output.squeeze(0), attn_applied.squeeze(0)), 1)  # (1, hidden_size * 2)
        output = self.out(output)  # (1, output_size)
        output = F.log_softmax(output, dim=1)

        return output, hidden, attn_weights.squeeze(0)


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, nhead=4, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Linear layer to match decoder's expected hidden size
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_sequence, hidden=None):
        # input_sequence: (seq_len, batch_size=1)

        # Embedding
        embedded = self.embedding(input_sequence)  # (seq_len, 1, hidden_size)
        embedded = embedded.squeeze(1)  # (seq_len, hidden_size)

        # Add positional encoding
        embedded = self.positional_encoding(embedded)

        # Transformer expects (seq_len, batch_size, hidden_size)
        embedded = embedded.unsqueeze(1)  # (seq_len, 1, hidden_size)

        # Transformer encoder forward pass
        transformer_output = self.transformer_encoder(embedded)  # (seq_len, 1, hidden_size)

        # Take mean of all hidden representations as sentence representation
        sentence_representation = transformer_output.mean(dim=0)  # (1, hidden_size)

        # Process through linear layer to match expected dimensions
        sentence_representation = self.fc(sentence_representation)  # (1, hidden_size)

        # For compatibility with existing code, return:
        # - Last token's output (like RNN)
        # - Sentence representation as "hidden state"
        return transformer_output[-1].unsqueeze(0), sentence_representation.unsqueeze(0)

    def initHidden(self):
        # Return dummy hidden state for compatibility
        return torch.zeros(1, 1, self.hidden_size, device=device)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
