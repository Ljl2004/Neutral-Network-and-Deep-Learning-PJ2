"""
Data loaders for CIFAR-10 with data augmentation
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_cifar_loader(root='./data/', batch_size=128, train=True,
                     shuffle=True, num_workers=4, augmentation=False):
    """Get CIFAR-10 data loader with optional data augmentation.

    Args:
        root: data directory
        batch_size: batch size
        train: whether to load training or test set
        shuffle: whether to shuffle the data
        num_workers: number of data loading workers
        augmentation: whether to apply data augmentation (training only)
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )

    if train and augmentation:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = datasets.CIFAR10(root=root, train=train, download=True,
                                   transform=train_transforms)
    else:
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        dataset = datasets.CIFAR10(root=root, train=train, download=True,
                                   transform=test_transforms)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True)
    return loader
