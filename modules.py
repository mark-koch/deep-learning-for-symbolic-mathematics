import torch
import torch.nn as nn
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def forward(self, V, K, Q, mask=None):
        """
        Args:
            V: Values of shape [batch_size, seq_length_v, d_model]. Must have same dimensionality as K, i.e.
                seq_length_v = seq_length_k
            K: Keys of shape [batch_size, seq_length_k, d_model]
            Q: Queries of shape [batch_size, seq_length_q, d_model]
            mask: Mask of entries to ignore in the attention matrix. In general padding positions should be masked out.
                For the decoder, positions in front of the current generation index should also be ignored. Masked out
                positions correspond to ones in the matrix. All other values should be zero.
                Shape depends on the type of mask, but must be broadcastable to [..., seq_length_q, seq_length_k].

        Returns:
            output: Shape [batch_size, seq_length_q, d_model]
            att_matrix: Attention matrix of shape [batch_size, num_heads, seq_length_q, seq_length_k]
        """
        V = self.linear_q(V)  # [batch_size, seq_length_v, d_model]
        K = self.linear_q(K)  # [batch_size, seq_length_k, d_model]
        Q = self.linear_q(Q)  # [batch_size, seq_length_q, d_model]

        # Split into multiple heads. Shape [batch_size, num_heads, seq_length, d_model / num_heads]
        batch_size = Q.shape[0]
        V = V.reshape([batch_size, -1, self.num_heads, self.d_model // self.num_heads]).transpose(1, 2)  # seq_len_k
        K = K.reshape([batch_size, -1, self.num_heads, self.d_model // self.num_heads]).transpose(1, 2)  # seq_len_k
        Q = Q.reshape([batch_size, -1, self.num_heads, self.d_model // self.num_heads]).transpose(1, 2)  # seq_len_q

        # Attention of shape [batch_size, num_heads, seq_length_q, seq_length_k]
        att_matrix = Q.matmul(K.transpose(2, 3) / np.sqrt(self.d_model // self.num_heads))
        if mask is not None:
            att_matrix += mask * -1e10  # Make masked out values really small, so that softmax turns it into 0
        att_matrix = att_matrix.softmax(dim=-1)  # [batch_size, num_heads, seq_length_q, seq_length_k]
        output = att_matrix.matmul(V)            # [batch_size, num_heads, seq_length_q, d_model / num_heads]

        # Concatenate attention heads again
        output = output.reshape([batch_size, -1, self.d_model])  # [batch_size, seq_length_q, d_model]

        output = self.linear(output)  # [batch_size, seq_length_q, d_model]

        return output, att_matrix


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-12)
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, x, padding_mask):
        """
        Args:
            x: Input of shape [batch_size, seq_len_dec, d_model]
            padding_mask: Mask that covers the padding characters in the encoder input.
                Shape [batch_size, 1, 1, seq_len]

        Returns:
            y: Output of shape [batch_size, seq_len, d_model]
            att_matrix: Attention matrix of shape [batch_size, num_heads, seq_len, seq_len]
        """
        att_out, att_matrix = self.mha(x, x, x, padding_mask)  # [batch_size, seq_length, d_model]
        att_out = self.dropout_1(att_out)
        att_out = self.layer_norm_1(att_out + x)

        y = self.linear_1(att_out)
        y = y.relu()
        y = self.linear_2(y)
        y = self.dropout_2(y)
        y = self.layer_norm_2(y + att_out)

        return y, att_matrix


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.mha_1 = MultiHeadAttention(d_model, num_heads)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-12)
        self.mha_2 = MultiHeadAttention(d_model, num_heads)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-12)
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout_3 = nn.Dropout(dropout_rate)
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, x, encoder_output, enc_padding_mask, dec_combined_mask):
        """
        Args:
            x: Input of shape [batch_size, seq_len_dec, d_model]
            encoder_output: Output by the encoder network. Shape [batch_size, seq_len_enc, d_model]
            enc_padding_mask: Mask that covers the padding characters in the encoder (!) input. Used mask out these
                padding positions in the second attention block. Shape [batch_size, 1, 1, seq_len_enc]
            dec_combined_mask: Mask that covers the padding characters in the encoder. Also covers the future tokens in
                the sequence. Used in the first attention block. Shape [batch_size, 1, seq_len_dec, seq_len_dec]

        Returns:
            y: Output of shape [batch_size, seq_len_dec, d_model]
            att_matrix_1, att_matrix_2: Attention matrices for both attention blocks with shapes [batch_size, num_heads,
                seq_len_dec, seq_len_dec]
        """
        # Shape of att_out_1: [batch_size, seq_len_dec, d_model]
        # Shape of att_matrix_1: [batch_size, seq_len_dec, seq_len_dec]
        att_out_1, att_matrix_1 = self.mha_1(x, x, x, dec_combined_mask)
        att_out_1 = self.dropout_1(att_out_1)
        att_out_1 = self.layer_norm_1(att_out_1 + x)

        att_out_2, att_matrix_2 = self.mha_2(encoder_output, encoder_output, x, enc_padding_mask)
        att_out_2 = self.dropout_2(att_out_2)
        att_out_2 = self.layer_norm_2(att_out_2 + att_out_1)

        y = self.linear_1(att_out_2)
        y = y.relu()
        y = self.linear_2(y)
        y = self.dropout_3(y)
        y = self.layer_norm_2(y + att_out_2)

        return y, att_matrix_1, att_matrix_2


class Encoder(nn.Module):
    def __init__(self, layers, d_model, num_heads, d_ff, dropout_rate, max_seq_len, vocab_size, vocab_padding_index=0):
        super(Encoder, self).__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=vocab_padding_index)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList()
        self.layers.extend([EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(layers)])

        # Create positional encoding lookup matrix. Must be registered as buffer, so that it gets put on the right
        # device later.
        self.register_buffer("positional_encoding", torch.zeros([1, max_seq_len, d_model], requires_grad=False))
        pos = torch.arange(max_seq_len)[:, np.newaxis]
        arg = pos * np.power(10000, 2 * torch.arange(d_model)[np.newaxis] / float(d_model))
        self.positional_encoding[0, :, 0::2] = torch.sin(arg[:, 0::2])
        self.positional_encoding[0, :, 1::2] = torch.cos(arg[:, 1::2])
        self.register_buffer("positional_encoding", self.positional_encoding)

    def forward(self, x, padding_mask):
        """
        Args:
            x: Input of shape [batch_size, seq_len_dec, d_model]
            padding_mask: Mask that covers the padding characters in the encoder input. Shape [batch, 1, 1, seq_len]

        Returns:
            y: Output of shape [batch_size, seq_len, d_model]
            attention: Dictionary containing the attention matrices. Shapes are [batch_size, num_heads, seq_len_dec,
                seq_len_dec]
        """
        seq_length = x.shape[1]
        attention = {}
        x = self.embedding(x) * np.sqrt(float(self.d_model))  # [batch_size, seq_length, d_model]
        x += self.positional_encoding[:, :seq_length, :]
        x = self.dropout(x)
        for i, layer in enumerate(self.layers):
            x, att = layer(x, padding_mask)
            attention["encoder_layer" + str(i+1)] = att
        return x, attention


class Decoder(nn.Module):
    def __init__(self, layers, d_model, num_heads, d_ff, dropout_rate, max_seq_len, vocab_size, vocab_padding_index=0):
        super(Decoder, self).__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=vocab_padding_index)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList()
        self.layers.extend([DecoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(layers)])

        # Create positional encoding lookup matrix. Must be registered as buffer, so that it gets put on the right
        # device later.
        self.register_buffer("positional_encoding", torch.zeros([1, max_seq_len, d_model], requires_grad=False))
        pos = torch.arange(max_seq_len)[:, np.newaxis]
        arg = pos * np.power(10000, 2 * torch.arange(d_model)[np.newaxis] / float(d_model))
        self.positional_encoding[0, :, 0::2] = torch.sin(arg[:, 0::2])
        self.positional_encoding[0, :, 1::2] = torch.cos(arg[:, 1::2])

    def forward(self, x, encoder_output, enc_padding_mask, dec_combined_mask):
        """
        Args:
            x: Input of shape [batch_size, seq_len_dec, d_model]
            encoder_output: Output by the encoder network. Shape [batch_size, seq_len_enc, d_model]
            enc_padding_mask: Mask that covers the padding characters in the encoder (!) input. Used mask out these
                padding positions in the second attention block. Shape [batch_size, 1, 1, seq_len_enc]
            dec_combined_mask: Mask that covers the padding characters in the encoder. Also covers the future tokens in
                the sequence. Used in the first attention block. Shape [batch_size, 1, seq_len_dec, seq_len_dec]

        Returns:
            y: Output of shape [batch_size, seq_len_dec, d_model]
            attention: Dictionary containing the attention matrices. Shapes are [batch_size, num_heads, seq_len_dec,
                seq_len_dec]
        """
        seq_length = x.shape[1]
        attention = {}
        x = self.embedding(x) * np.sqrt(float(self.d_model))  # [batch_size, seq_length, d_model]
        x += self.positional_encoding[:, :seq_length, :]
        x = self.dropout(x)
        for i, layer in enumerate(self.layers):
            x, att_1, att_2 = layer(x, encoder_output, enc_padding_mask, dec_combined_mask)
            attention["decoder_layer{}_block1".format(i+1)] = att_1
            attention["decoder_layer{}_block2".format(i+1)] = att_2
        return x, attention


class Transformer(nn.Module):
    """
    Args:
        encoder_layers: Number of layers for the encoder
        decoder_layers: Number of layers for the decoder
        d_model: Dimensionality of the model
        num_heads: Number of heads for multi-head attention
        d_ff: Dimensionality of inner layer for the feed forward networks in each layer
        dropout_rate: Probability for dropout
        max_input_seq_len: Maximum length of the input sequence that will be provided to the encoder
        max_target_seq_len: Maximum length of the target sequence. These sequences will be fed to the decoder.
        input_vocab_size: Number of characters in the input vocab.
        target_vocab_size: Number of characters in the target vocab.
        vocab_padding_index: Index of the padding char of the vocab. This should be the same for input and target vocab.
    """
    def __init__(self, encoder_layers, decoder_layers, d_model, num_heads, d_ff, dropout_rate, max_input_seq_len,
                 max_target_seq_len, input_vocab_size, target_vocab_size, vocab_padding_index=0):
        super(Transformer, self).__init__()
        self.num_encoder_layers = encoder_layers
        self.num_decoder_layers = decoder_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.encoder = Encoder(encoder_layers, d_model, num_heads, d_ff, dropout_rate, max_input_seq_len,
                               input_vocab_size, vocab_padding_index)
        self.decoder = Decoder(decoder_layers, d_model, num_heads, d_ff, dropout_rate, max_target_seq_len,
                               target_vocab_size, vocab_padding_index)
        self.linear = nn.Linear(d_model, target_vocab_size, bias=True)

    def forward(self, encoder_input, decoder_input, enc_padding_mask, dec_combined_mask):
        """
        Args:
            encoder_input: Input sequence to feed to the encoder. Shape [batch_size, seq_len_enc, d_model]
            decoder_input: Target sequence to feed to the encoder. Shape [batch_size, seq_len_dec, d_model]
            enc_padding_mask: Mask that covers the padding characters in the encoder (!) input. Used mask out these
                padding positions in the second attention block. Shape [batch_size, 1, 1, seq_len_enc]
            dec_combined_mask: Mask that covers the padding characters in the encoder. Also covers the future tokens in
                the sequence. Used in the first attention block. Shape [batch_size, 1, seq_len_dec, seq_len_dec]

        Returns:
            y_logits, y: Logits and softmax output of shape [batch_size, seq_len_dec, target_vocab_size]
            attention: Dictionary containing the attention matrices. Shapes are [batch_size, num_heads, seq_len_dec,
                seq_len_dec]
        """
        enc_output, enc_att = self.encoder(encoder_input, enc_padding_mask)  # [batch_size, seq_length_enc, d_model]
        dec_output, dec_att = self.decoder(decoder_input, enc_output, enc_padding_mask, dec_combined_mask)  # [...]

        y_logits = self.linear(dec_output)  # [batch_size, seq_len_dec, target_vocab_size]
        y = y_logits.softmax(dim=-1)
        return y_logits, y, {**enc_att, **dec_att}

