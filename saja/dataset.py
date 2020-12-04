import dataclasses
import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import uproot


def get_data_mask(data, length):
    mask_shape = data.shape[:-1]
    data_mask = torch.full(size=mask_shape, fill_value=False, dtype=torch.bool)
    for m, l in zip(data_mask, length):
        m[: l].fill_(True)
    return data_mask


@dataclasses.dataclass
class Batch:
    data: Tensor
    target: Tensor
    length: Tensor
    mask: Tensor

    def to(self, device):
        return Batch(*[each.to(device) for each in dataclasses.astuple(self)])


class SaJaDataset(torch.utils.data.Dataset):

    def __init__(self, path, treepath, data_branches, target_branch):
        self.path = path
        self.treepath = treepath
        self.data_branches = data_branches
        self.target_branch = target_branch

        branches = data_branches + [target_branch]
        tree_iter = uproot.iterate(path, treepath, branches=branches,
                                   namedecode='utf-8')

        self._examples = []
        for chunk in tree_iter:
            self._examples += self._process(chunk)

    def _process(self, chunk):
        data_chunk = [chunk[branch] for branch in self.data_branches]
        data_chunk = zip(*data_chunk)
        data_chunk = [np.stack(each, axis=1) for each in data_chunk]
        data_chunk = [each.astype(np.float32) for each in data_chunk]
        data_chunk = [torch.from_numpy(each) for each in data_chunk]

        target_chunk = chunk[self.target_branch]
        target_chunk = [each.astype(np.int64) for each in target_chunk]
        target_chunk = [torch.from_numpy(each) for each in target_chunk]

        example_chunk = list(zip(data_chunk, target_chunk))
        return example_chunk

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self._examples[idx]
 
    @classmethod
    def collate(cls, batch):
        data, target = list(zip(*batch))
        length = torch.LongTensor([each.size(0) for each in data])
        data = pad_sequence(data, batch_first=True)
        mask = get_data_mask(data, length)
        target = pad_sequence(target, batch_first=True)
        return Batch(data, target, length, mask)
