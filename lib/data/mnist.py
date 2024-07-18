import torch
from torchvision import datasets, transforms
from lib.data.utils import _one_hot_ten

def create_mnist_loaders(batch_size):
    """
    Create dataloaders for the training and test set of MNIST.

    Args:
        batch_size: Number of samples per batch

    Returns:
        train_loader: torch.utils.data.DataLoader for the MNIST training set
        val_loader: torch.utils.data.DataLoader for the MNIST validation set
        test_loader: torch.utils.data.DataLoader for the MNIST test set
    """

    # Load train and test MNIST datasets
    mnist_train = datasets.MNIST('../data/', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,)),
                                 ]),
                                 target_transform=_one_hot_ten
                                 )

    # Define the sizes of the training and validation sets
    train_size = int(0.9 * len(mnist_train))
    val_size = len(mnist_train) - train_size

    # Split the dataset
    mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [train_size, val_size])

    mnist_test = datasets.MNIST('../data/', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                ]),
                                target_transform=_one_hot_ten
                                )

    # For GPU acceleration store dataloader in pinned (page-locked) memory
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # Create the dataloader objects
    train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, drop_last=True, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        mnist_val, batch_size=batch_size, drop_last=True, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, drop_last=True, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader
