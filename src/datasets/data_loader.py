from pathlib import Path
from torch.utils.data import DataLoader
from sisfall_dataset import SisFallDataset

def get_dataloaders(cfg):
    train = SisFallDataset("train", Path(cfg.data.processed_dir))
    val   = SisFallDataset("val", Path(cfg.data.processed_dir))
    test  = SisFallDataset("test", Path(cfg.data.processed_dir))

    train_loader = DataLoader(train, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader   = DataLoader(val, batch_size=cfg.data.batch_size)
    test_loader  = DataLoader(test, batch_size=cfg.data.batch_size)
    return train_loader, val_loader, test_loader
