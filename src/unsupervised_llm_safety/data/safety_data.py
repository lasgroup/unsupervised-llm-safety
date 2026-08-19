import torch
from torch.utils.data import DataLoader, Dataset


class HiddenStatesDataset(Dataset):
    def __init__(self, data):
        self.data = data["hidden_states"]
        self.label = data["labels"]

        # Create mapping from unique paths to indices
        unique_sources = list(set(data["data_source"]))
        self.source_to_idx = {
            source: idx for idx, source in enumerate(unique_sources)
        }
        self.idx_to_source = {
            idx: source for source, idx in self.source_to_idx.items()
        }

        # Convert data_source to tensor of indices
        self.data_source = torch.tensor(
            [self.source_to_idx[source] for source in data["data_source"]],
            dtype=torch.long,
        )

        self.data = self.data.to(dtype=torch.float32)
        self.label = self.label.to(dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], idx


def get_hidden_states_dataloader(
    hidden_states_data, batch_size=32, shuffle=True, num_workers=1
):
    dataloader = DataLoader(
        HiddenStatesDataset(hidden_states_data),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader
