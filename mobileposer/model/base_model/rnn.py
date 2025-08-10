import torch.nn as nn
from torch.nn.functional import relu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import *
import articulate as art

class RNNLossWrapper:
    r"""
    Loss wrapper for `articulate.utils.torch.RNN`.
    """
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, y_pred, y_true):
        return self.loss_fn(torch.cat(y_pred), torch.cat(y_true))

class RNNDataset(torch.utils.data.Dataset):
    r"""
    Dataset for `articulate.utils.torch.RNN`.
    """
    def __init__(self, data: list, label: list, split_size=-1, augment_fn=None, device=None, drop_last=False):
        r"""
        Init an RNN dataset.

        Notes
        -----
        Get the dataloader by torch.utils.data.DataLoader(dataset, **collate_fn=RNNDataset.collate_fn**)

        If `split_size` is positive, `data` and `label` will be split to lists of small sequences whose lengths
        are not larger than `split_size`. Otherwise, it has no effects.

        If `augment_fn` is not None, `data` item will be augmented like `augment_fn(data[i])` in `__getitem__`.
        Otherwise, it has no effects.

        Args
        -----
        :param data: A list that contains sequences(tensors) in shape [num_frames, n_input].
        :param label: A list that contains sequences(tensors) in shape [num_frames, n_output].
        :param split_size: If positive, data and label will be split to list of small sequences.
        :param augment_fn: If not None, data item will be augmented in __getitem__.
        :param device: The loaded data is finally copied to the device. If None, the device of data[0] is used.
        :param drop_last: Whether to drop the last element during splitting (if not in full size).
        """
        assert len(data) == len(label) and len(data) != 0
        if split_size > 0:
            self.data, self.label = [], []
            if drop_last:
                for td, tl in zip(data, label):
                    if td.shape[0] % split_size != 0:
                        self.data.extend(td.split(split_size)[:-1])
                        self.label.extend(tl.split(split_size)[:-1])
                        if td.shape[0] > split_size:
                            self.data.append(td[-split_size:])
                            self.label.append(tl[-split_size:])
                    else:
                        self.data.extend(td.split(split_size))
                        self.label.extend(tl.split(split_size))
            else:
                for td, tl in zip(data, label):
                    self.data.extend(td.split(split_size))
                    self.label.extend(tl.split(split_size))
        else:
            self.data = data
            self.label = label
        self.augment_fn = augment_fn
        self.device = device if device is not None else data[0].device

    def __getitem__(self, i):
        data = self.data[i] if self.augment_fn is None else self.augment_fn(self.data[i])
        label = self.label[i]
        return data.to(self.device), label.to(self.device)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(x):
        r"""
        [[seq0, label0], [seq1, label1], [seq2, label2]] -> [[seq0, seq1, seq2], [label0, label1, label2]]
        """
        return list(zip(*x))

class RNNWithInitDataset(RNNDataset):
    r"""
    The same as `RNNDataset`. Used for `RNNWithInit`.
    """
    def __init__(self, data: list, label: list, split_size=-1, augment_fn=None, device=None, drop_last=False):
        super(RNNWithInitDataset, self).__init__(data, label, split_size, augment_fn, device, drop_last)

    def __getitem__(self, i):
        data, label = super(RNNWithInitDataset, self).__getitem__(i)
        return (data, label[0]), label

class RNN(torch.nn.Module):
    """
    A RNN Module including a linear input layer, a RNN and a linear output layer.
    """
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.4):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(n_hidden, n_hidden, num_layers=n_rnn_layer, bidirectional=bidirectional)
        self.linear1 = nn.Linear(in_features=n_input, out_features=n_hidden)
        self.linear2 = nn.Linear(in_features=n_hidden * (2 if bidirectional else 1), out_features=n_output)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, seq_lengths=None, h=None, mean_output=False):
        # pass input data through a linear layer
        data = self.dropout(relu(self.linear1(x)))
        # pack the padded sequences
        if seq_lengths is not None:
            data = pack_padded_sequence(data, seq_lengths, batch_first=True, enforce_sorted=False)
        # pass input to RNN
        data, h = self.rnn(data, h)
        
        if mean_output:
            # if mean_output is True, return the mean of the output
            data = data.mean(dim=1, keepdim=False)
        
        # pack the padded sequences
        output_lengths = None
        if seq_lengths is not None:
            data, output_lengths = pad_packed_sequence(data, batch_first=True)
        data = self.linear2(data)
        return data, output_lengths, h

class RNNWithInit(RNN):
    r"""
    RNN with the initial hidden states regressed from the first output.
    """
    def __init__(self, n_input: int, n_output: int, n_hidden: int, n_rnn_layer: int, init_size: int=None,
                 bidirectional=False, dropout=0., layer_norm=False,
                 rnn_type='lstm', load_weight_file: str = None):
        r"""
        Init an RNNWithInit net.
        
        :param n_input: Input size.
        :param n_output: Output size.
        :param n_hidden: Hidden size for RNN.
        :param n_rnn_layer: Number of RNN layers.
        :param init_size: Init net size. Default output size.
        :param rnn_type: Select from 'rnn', 'lstm', 'lnlstm', 'gru'.
        :param bidirectional: Whether if the RNN is bidirectional.
        :param input_linear: Whether to apply a Linear layer (n_input, n_hidden) to the input.
        :param same_sequence_length: Whether are the input sequence lengths the same.
        :param dropout: Dropout after the input linear layer and in the rnn.
        :param layer_norm: Whether to apply layer norm to h and c.
        :param load_weight_file: If not None and exists, weights will be loaded.
        """
        assert rnn_type.upper() == 'LSTM' or rnn_type.upper() == 'LNLSTM' and bidirectional is False
        super().__init__(n_input, n_output, n_hidden, n_rnn_layer, bidirectional, dropout)
        self.num_layers = n_rnn_layer
        self.bidirectional = bidirectional
        self.n_hidden = n_hidden

        self.init_net = torch.nn.Sequential(
            torch.nn.Linear(init_size or n_output, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden * n_rnn_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden * n_rnn_layer, 2 * (2 if bidirectional else 1) * n_rnn_layer * n_hidden),
            torch.nn.LayerNorm(2 * (2 if bidirectional else 1) * n_rnn_layer * n_hidden) if layer_norm else torch.nn.Identity()
        )

        if load_weight_file:
            self.load_state_dict(torch.load(load_weight_file, map_location=torch.device('cpu')))
            self.eval()

    def forward(self, x, seq_lengths=None):
        r"""
        Forward.

        :param x: A list in length [batch_size] which contains 2-tuple
                  (Tensor[num_frames, n_input], Tensor[n_output]).
        :param _: Not used.
        :return: A list in length [batch_size] which contains tensors in shape [num_frames, n_output].
        """
        # x, x_init = list(zip(*x))
        x, x_init = x
        nd, nh = self.num_layers * (2 if self.bidirectional else 1), self.n_hidden
        # h, c = self.init_net(torch.stack(x_init)).view(-1, 2, nd, nh).permute(1, 2, 0, 3)
        h, c = self.init_net(x_init).view(-1, 2, nd, nh).permute(1, 2, 0, 3)
        return super(RNNWithInit, self).forward(x, h=(h, c), seq_lengths=seq_lengths)