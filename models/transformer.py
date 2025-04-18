import torch
import torch.nn as nn

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
