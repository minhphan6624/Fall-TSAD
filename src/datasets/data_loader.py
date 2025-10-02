import torch
from torch.utils.data import Dataset, DataLoader
from src.datasets.sisfall_serialize import load_processed_data

class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for loading time series data.
    """
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_dataloaders(processed_data_path, batch_size):
    # Load processed data
    processed_data = load_processed_data(processed_data_path)

    train_data = processed_data['train_data']
    train_labels = processed_data['train_labels']
    val_data = processed_data['val_data']
    val_labels = processed_data['val_labels']
    test_data = processed_data['test_data']
    test_labels = processed_data['test_labels']

    # Create Datasets
    train_dataset = TimeSeriesDataset(train_data, train_labels)
    val_dataset = TimeSeriesDataset(val_data, val_labels)
    test_dataset = TimeSeriesDataset(test_data, test_labels)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
