import torch

def _one_hot_ten(label, num_classes=10):
    """
    Helper function to convert to a one hot encoding with **num_classes** classes.

    Args:
        label: target label as single number
        num_classes (int): number of classes 

    Returns:
        One-hot tensor with dimension (*, num_classes) encoding label
    """
    return torch.nn.functional.one_hot(torch.tensor(label), num_classes)
