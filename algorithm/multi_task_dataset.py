from torch.utils.data import Dataset, DataLoader, random_split

class MultiTaskDataset(Dataset):
    def __init__(self, data, targets, masks, event_all):
        self.data = data
        self.targets = targets
        self.masks = masks
        self.event_all = event_all

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], [self.targets[i][idx] for i in range(len(self.targets))], [self.masks[i][idx] for i in range(len(self.masks))], self.event_all[idx]
