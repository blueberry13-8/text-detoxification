from torch import nn
import torch


class Transformer(nn.Module):
    def __init__(
            self,
            embedding_size,
            vocab_size,
            src_pad_idx,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dropout,
            device
    ):
        """
        Initialize the Transformer model.

        Args:
            embedding_size (int): The embedding dimension.
            vocab_size (int): The size of the vocabulary.
            src_pad_idx (int): The index of padding in the source sequences.
            num_heads (int): The number of attention heads in the transformer.
            num_encoder_layers (int): The number of encoder layers.
            num_decoder_layers (int): The number of decoder layers.
            dropout (float): The dropout probability.
            device (str): The device to run the model on (e.g., 'cpu' or 'cuda').
        """
        super(Transformer, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(50, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(embedding_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        """
        Create a source mask to identify padding in the source sequences.

        Args:
            src (torch.Tensor): The source sequence tensor.

        Returns:
            src_mask (torch.Tensor): The source padding mask.
        """
        src_mask = src == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        """
        Forward pass of the Transformer model.

        Args:
            src (torch.Tensor): The source sequence tensor.
            trg (torch.Tensor): The target sequence tensor.

        Returns:
            out (torch.Tensor): The model's output sequence.
        """
        N, src_seq_length = src.shape
        N, trg_seq_length = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
                .unsqueeze(0)
                .expand(N, src_seq_length)
                .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
                .unsqueeze(0)
                .expand(N, trg_seq_length)
                .to(self.device)
        )

        embed_src = self.dropout(
            (self.word_embedding(src) + self.position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.word_embedding(trg) + self.position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out
