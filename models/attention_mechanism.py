import torch
import torch.nn as nn
import torch.nn.functional as F

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
