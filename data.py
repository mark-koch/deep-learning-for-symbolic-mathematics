import torch.utils.data
from torch.utils.data.dataset import IterableDataset
from vocab import token2id


def parse_paper_format(line):
    line = line.split("|")[1]
    f, F = line.split("\t")
    f = [token2id(t.strip()) for t in f.split(" ")[2:]]
    F = [token2id(t.strip()) for t in F.split(" ")]
    return f, F


class SymbolicIntegrationDataset(IterableDataset):
    """
    Symbolic integration dataset.

    Because of the large data files, this dataset doesn't keep the data in memory. Instead we read the files line by
    line and alternately yield data from each file. Note that for this reason we cannot shuffle the data. Also this
    class is set up to work with only a single worker. It does not support multiprocessing.
    """
    def __init__(self, data_paths, device, max_input_len=-1, max_target_len=-1):
        self.data_paths = data_paths
        self.device = device
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.files = []
        self.current_file = 0
        for path in data_paths:
            self.files.append(open(path, "r", encoding="utf-8"))

    def __iter__(self):
        while True:
            try:
                line = next(self.files[self.current_file])
                try:
                    inp, tar = parse_paper_format(line)
                except Exception:
                    print("WARNING: Couldn't read line '{}'".format(line))
                    continue
                inp.insert(0, token2id("<sos>"))
                tar.insert(0, token2id("<sos>"))
                inp.append(token2id("<eos>"))
                tar.append(token2id("<eos>"))
                if (len(inp) > self.max_input_len > 0) or (len(tar) > self.max_target_len > 0):
                    continue
                yield torch.tensor(inp, device=self.device), torch.tensor(tar, device=self.device)
                if len(self.data_paths) > 1:
                    self.current_file = (self.current_file + 1) % (len(self.data_paths) - 1)
            except StopIteration:
                # The file has ended. We need to reopen the stream
                self.files[self.current_file].close()
                self.files[self.current_file] = open(self.data_paths[self.current_file], "r", encoding="utf-8")

    def seek_forward(self, num_lines):
        """
        Skips the first num_lines of the data (wraps around at the end). Can be used to continue training at the
        previous position in the dataset.
        Note that the input will not be parsed, it simply counts lines. This means lines that are too long, short or
        invalid (these will be skipped during training) will lead to slight shifts.
        """
        for _ in range(num_lines):
            try:
                next(self.files[self.current_file])
                if len(self.data_paths) > 1:
                    self.current_file = (self.current_file + 1) % (len(self.data_paths) - 1)
            except StopIteration:
                # The file has ended. We need to reopen the stream
                self.files[self.current_file].close()
                self.files[self.current_file] = open(self.data_paths[self.current_file], "r", encoding="utf-8")


def collate_fn(data):
    """
    Creates mini-batch tensors from (input, target) tuples.

    Args:
        data: Dictionary with padded training data and masks
    """
    max_input_len = max(d[0].shape[0] for d in data)
    max_target_len = max(d[1].shape[0] for d in data)
    batch_size = len(data)

    # Prepare zero padding
    input_padded = torch.zeros(batch_size, max_input_len, dtype=torch.long, requires_grad=False)
    target_padded = torch.zeros(batch_size, max_target_len, dtype=torch.long, requires_grad=False)

    # Padding masks of shape [batch_size, 1, 1, seq_len]. Empty dimensions are needed for broadcasting in multi-head
    # attention to [..., seq_len_q, seq_len_k].
    enc_padding_mask = torch.ones(batch_size, 1, 1, max_input_len, dtype=torch.bool, requires_grad=False)
    dec_padding_mask = torch.ones(batch_size, 1, 1, max_target_len, dtype=torch.bool, requires_grad=False)

    for i, d in enumerate(data):
        input_padded[i, :d[0].shape[0]] = d[0]
        target_padded[i, :d[1].shape[0]] = d[1]
        enc_padding_mask[i, 0, 0, :d[0].shape[0]] = 0
        dec_padding_mask[i, 0, 0, :d[1].shape[0]] = 0

    # For each position in the decoder sequence, mask out the positions that are in the future. The resulting matrix
    # looks like this
    #                                          [[0, 1, 1, 1],
    #                                           [0, 0, 1, 1],
    #                                           [0, 0, 0, 1],
    #                                           [0, 0, 0, 0]]
    look_ahead_mask = torch.ones(max_target_len, max_target_len, dtype=torch.bool, requires_grad=False).triu(1).bool()

    dec_combined_mask = dec_padding_mask | look_ahead_mask
    dec_combined_mask.requires_grad = False

    return {"input": input_padded, "target": target_padded, "enc_padding_mask": enc_padding_mask,
            "dec_combined_mask": dec_combined_mask, "dec_padding_mask": dec_padding_mask}


def get_validation_batches(data_path, batch_size, max_input_len=-1, max_target_len=-1):
    """
    Returns a list of dictionaries with padded validation data and masks for each batch.
    """
    batches = []
    current_batch = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            inp, tar = parse_paper_format(line)
            inp.insert(0, token2id("<sos>"))
            tar.insert(0, token2id("<sos>"))
            inp.append(token2id("<eos>"))
            tar.append(token2id("<eos>"))
            if (len(inp) > max_input_len > 0) or (len(tar) > max_target_len > 0):
                continue
            current_batch.append((torch.tensor(inp), torch.tensor(tar)))

            if len(current_batch) == batch_size:
                batches.append(collate_fn(current_batch))
                current_batch = []
    # Final batch may have less than batch_size entries
    if len(current_batch) > 0:
        batches.append(collate_fn(current_batch))

    return batches

