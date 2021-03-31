import dataclasses
from typing import Union
from typing import List
import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import tqdm
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


# TODO typing
class JetPartonAssignmentDataset(torch.utils.data.Dataset):

    def __init__(self,
                 path: Union[str, List[str]],
                 treepath: str,
                 data_branches: List[str],
                 target_branch: str,
                 num_workers: int = 1,
    ) -> None:
        """
        Args:
        """
        self.data_branches = data_branches
        self.target_branch = target_branch

        files = self._to_files(path, treepath)
        branches = data_branches + [target_branch]

        tree_iter = uproot.iterate(files, expressions=branches, library='np',
                                   num_workers=num_workers)

        total = self._get_total_entries(files)
        pbar = tqdm.tqdm(tree_iter)
        self._examples = []

        def print_progress():
            processed = len(self._examples)
            pbar.set_description(f'Total = {total:d}, Processed: {processed:d}'
                                 f' ({100 * processed / total:.2f} %)')

        print_progress()
        for chunk in tree_iter:
            self._examples += self._process(chunk)
            print_progress()

    def _to_files(self, files, treepath):
        if isinstance(files, str):
            files = {files: treepath}
        elif isinstance(files, list):
            files = {each: treepath for each in files}
        else:
            raise TypeError
        return files

    def _get_total_entries(self, files) -> int:
        num_entries = 0
        for path, treepath in files.items():
            root_file = uproot.open(path)
            tree = root_file[treepath]
            num_entries += tree.num_entries
        return num_entries


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
