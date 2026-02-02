from pathlib import Path
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader

DATA_DIR = "data/cifar-10-batches-py"
SETS = ["train", "valid", "test"]

class CIFAR10Loader:
    def __init__(self, batch_size: int, normalize):
        self.name = 'CIFAR10'
        self.data_dir = Path(DATA_DIR)
        num_workers = 8
        transform = transforms.Compose([transforms.ToTensor(), normalize])

        whole_tset = CIFAR10(root=self.data_dir.parent, train=True, download=True,
                             transform=transform)
        testset = CIFAR10(root=self.data_dir.parent, train=False, download=True, 
                          transform=transform)

        perm = torch.randperm(len(whole_tset))
        val_len = int(len(perm)*0.1) # 10% for validation
        trainset = Subset(whole_tset, perm[val_len:])
        validset = Subset(whole_tset, perm[:val_len])

        self.train = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)
        self.valid = DataLoader(validset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers)
        self.test = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers)

        self.batch_size = batch_size

        self.in_chan = 3
        self.in_size = (32, 32)
        self.out_dim = 10

    def get_tensors(self, sets):
        for i, dset in enumerate(SETS):
            images, labels = [], []
            for img, label in sets[i]:
                images.append(img)
                labels.append(torch.tensor(label))
            torch.save(torch.stack(images), self.data_dir/f"{dset}_tensors.pt") 
            torch.save(torch.stack(labels), self.data_dir/f"{dset}_labels.pt")

    def get_config(self):
        return {
                "task": self.name,
                "in_chan": self.in_chan,
                "in_size": self.in_size,
                "out_dim": self.out_dim,
                "train_samples": len(self.train.dataset),
                "valid_samples": len(self.valid.dataset),
                "test_samples": len(self.test.dataset),
                }
