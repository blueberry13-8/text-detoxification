import torch
from torch import nn
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        """
        Initialize the Encoder module.

        Args:
            input_dim (int): The input dimension (size of the input vocabulary).
            emb_dim (int): The embedding dimension.
            hid_dim (int): The hidden dimension of the LSTM layers.
            n_layers (int): The number of LSTM layers.
            dropout (float): The dropout probability.
        """
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Forward pass of the Encoder.

        Args:
            src (torch.Tensor): The source input tensor.

        Returns:
            hidden (torch.Tensor): The hidden states from the LSTM.
            cell (torch.Tensor): The cell states from the LSTM.
        """
        embedded = self.dropout(self.embedding(src))

        outputs, (hidden, cell) = self.rnn(embedded)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        """
        Initialize the Decoder module.

        Args:
            output_dim (int): The output dimension (size of the output vocabulary).
            emb_dim (int): The embedding dimension.
            hid_dim (int): The hidden dimension of the LSTM layers.
            n_layers (int): The number of LSTM layers.
            dropout (float): The dropout probability.
        """
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        """
        Forward pass of the Decoder.

        Args:
            input (torch.Tensor): The input tensor for a single time step (one word).
            hidden (torch.Tensor): The hidden state from the previous time step.
            cell (torch.Tensor): The cell state from the previous time step.

        Returns:
            prediction (torch.Tensor): The output prediction for the current time step.
            hidden (torch.Tensor): The hidden state for the current time step.
            cell (torch.Tensor): The cell state for the current time step.
        """
        # input = [batch size, 1]
        embedded = self.dropout(self.embedding(input))
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        prediction = self.fc_out(output)

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        """
        Initialize the Seq2Seq model.

        Args:
            encoder (nn.Module): The encoder module.
            decoder (nn.Module): The decoder module.
            device (str): The device to run the model on (e.g., 'cpu' or 'cuda').
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass of the Seq2Seq model.

        Args:
            src (torch.Tensor): The source input sequence tensor.
            trg (torch.Tensor): The target output sequence tensor.
            teacher_forcing_ratio (float): The probability of using teacher forcing during training.

        Returns:
            outputs (torch.Tensor): The model's output sequence.
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output[:, 0, :]
            input = trg[:, t] if random.random() < teacher_forcing_ratio else output.argmax(2).squeeze(1)
        return outputs
