import torch.utils.data
import torch.utils.data.dataset
import io
from vocab import token2id


def parse_paper_format(line):
    line = line.split("|")[1]
    f, F = line.split("\t")
    f = [token2id(t.strip()) for t in f.split(" ")[2:]]
    F = [token2id(t.strip()) for t in F.split(" ")]
    return f, F


class SymbolicIntegrationDataset(torch.utils.data.dataset.Dataset):
    """ Symbolic integration dataset. """
    def __init__(self, data_paths, max_input_len=-1, max_target_len=-1):
        self.data = []
        self.max_input_len = 0
        self.max_target_len = 0
        for path in data_paths:
            with io.open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    f, F = parse_paper_format(line)
                    f.insert(0, token2id("<sos>"))
                    F.insert(0, token2id("<sos>"))
                    f.append(token2id("<eos>"))
                    F.append(token2id("<eos>"))
                    if (len(f) > max_input_len > 0) or (len(f) > max_target_len > 0):
                        continue
                    self.max_input_len = max(self.max_input_len, len(f))
                    self.max_target_len = max(self.max_target_len, len(F))
                    self.data.append((torch.tensor(f), torch.tensor(F)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]


def collate_fn(data):
    """
    Creates mini-batch tensors from (f, F) tuples.

    Args:
        data: List of (f, F) tuples containing the training data.
    """
    max_input_len = max(d[0].shape[0] for d in data)
    max_target_len = max(d[1].shape[0] for d in data)
    batch_size = len(data)

    # Prepare zero padding
    input_padded = torch.zeros(batch_size, max_input_len, dtype=torch.long)
    target_padded = torch.zeros(batch_size, max_target_len, dtype=torch.long)

    # Padding masks of shape [batch_size, 1, 1, seq_len]. Empty dimensions are needed for broadcasting in multi-head
    # attention to [..., seq_len_q, seq_len_k].
    enc_padding_mask = torch.zeros(batch_size, 1, 1, max_input_len, dtype=torch.float)
    dec_padding_mask = torch.zeros(batch_size, 1, 1, max_target_len, dtype=torch.float)

    for i, d in enumerate(data):
        input_padded[i, :d[0].shape[0]] = d[0]
        target_padded[i, :d[1].shape[0]] = d[1]
        enc_padding_mask[i, 0, 0, :d[0].shape[0]] = 1
        dec_padding_mask[i, 0, 0, :d[1].shape[0]] = 1

    # For each position in the decoder dequence, mask out the positions that are in the future. The resulting matrix
    # looks like this
    #                                          [[0, 1, 1, 1],
    #                                           [0, 0, 1, 1],
    #                                           [0, 0, 0, 1],
    #                                           [0, 0, 0, 0]]
    look_ahead_mask = torch.ones(max_target_len, max_target_len, dtype=torch.float).triu(diagonal=1)

    dec_combined_mask = torch.max(dec_padding_mask, look_ahead_mask)

    return {"input": input_padded, "target": target_padded, "enc_padding_mask": enc_padding_mask,
            "dec_combined_mask": dec_combined_mask, "dec_padding_mask": dec_padding_mask}

