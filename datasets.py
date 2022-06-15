import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from functools import partial

def preprocess(x):
    n_bits = 8
    x = x * 255  # undo ToTensor scaling to [0,1]
    n_bins = 2 ** n_bits
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5
    return x

def postprocess(x):
    n_bits = 8
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    x = x * 2 ** n_bits
    return torch.clamp(x, 0, 255).byte()

def one_hot_encode(target, num_classes=10):
    """
    One hot encode with fixed 10 classes
    Args: target - the target labels to one-hot encode
    Retn: one_hot_encoding - the OHE of this tensor
    """
    one_hot_encoding = F.one_hot(torch.tensor(target), num_classes).float()
    return one_hot_encoding

def get_MNIST(num_classes):
    assert num_classes in [2, 10]    
    train_transform = transforms.Compose([
        transforms.Pad(2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.expand(3, -1, -1)),
        preprocess
    ])
    
    test_transform = transforms.Compose([
        transforms.Pad(2), 
        transforms.ToTensor(), 
        transforms.Lambda(lambda t: t.expand(3, -1, -1)), 
        preprocess
    ])
    
    train_dataset = datasets.MNIST(
        "./data/MNIST",
        train=True,
        transform=train_transform,
        download=True,
    )

    test_dataset = datasets.MNIST(
        "./data/MNIST",
        train=False,
        transform=test_transform,
        download=True,
    )
    
    if num_classes==2:
        train_idx = [torch.where(train_dataset.targets==c)[0] for c in [0, 1]]
        test_idx = [torch.where(test_dataset.targets==c)[0] for c in [0, 1]]
    
        train_idx = torch.cat(train_idx)
        test_idx = torch.cat(test_idx)
    
        train_dataset.targets = train_dataset.targets[train_idx]
        train_dataset.data = train_dataset.data[train_idx]
        
        test_dataset.targets = test_dataset.targets[test_idx]
        test_dataset.data = test_dataset.data[test_idx]
        
    train_dataset.target_transform = partial(one_hot_encode, num_classes=num_classes)
    test_dataset.target_transform = partial(one_hot_encode, num_classes=num_classes)
    return train_dataset, test_dataset