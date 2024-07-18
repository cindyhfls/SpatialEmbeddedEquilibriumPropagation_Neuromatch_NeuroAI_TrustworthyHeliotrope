import torch

def _one_hot_ten(label, num_classes:int=10):
    """
    Helper function to convert to a one hot encoding with **num_classes** classes.

    Args:
        label: target label as single number
        num_classes (int): number of classes 

    Returns:
        One-hot tensor with dimension (*, num_classes) encoding label
    """
    return torch.nn.functional.one_hot(torch.tensor(label), num_classes)

def get_random_sample_dataloader(dataset:torch.utils.data.Dataset, batch_size:int, M:int):
    """(From `W1D2_Tutorial1.ipynb`)

    Get a DataLoader for a subset of size **M** from **dataset** with batch size **batch_size**.

    WARNING:: 
    This is not per class.

    Args:
        dataset (torch.utils.data.Dataset): A dataset, can be obtained from a DataLoader: DataLoader.dataset
        batch_size (int): Batch size
        M (int): Number of samples

    Returns:
        (torch.utils.data.DataLoader): DataLoader for a subset of size **M** from **dataset** with batch size **batch_size**
    """
    indices = torch.randperm(len(dataset))[:M]
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    sampled_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return sampled_loader

def get_random_sample_train_val(train_dataset:torch.utils.data.Dataset, val_dataset:torch.utils.data.Dataset, batch_size:int, N_train_data:int):
    """(From `W1D2_Tutorial1.ipynb`)

    Get `DataLoader`s for randomly sampled subsets from **train_dataset** and **val_dataset**.

    Args:
        train_dataset (torch.utils.data.Dataset): A train dataset, can be obtained from a DataLoader: DataLoader.dataset
        val_dataset (torch.utils.data.Dataset): A validation dataset, can be obtained from a DataLoader: DataLoader.dataset
        batch_size (int): Batch size
        N_train_data (int): Number of samples in the train subset (~1/9 the number of samples in the validation subset, up to 30 for the validation subset)

    Returns:
        (torch.utils.data.DataLoader): train DataLoader for a subset of size **N_train_data** from **train_dataset** with batch size **batch_size**
        (torch.utils.data.DataLoader): DataLoader for a subset of size approximately **N_train_data**/9 (up to 30 max) from **val_dataset** with batch size **batch_size**
    """
    sampled_train_loader = get_random_sample_dataloader(train_dataset, batch_size, N_train_data)

    N_val_data = int(N_train_data / 9.0)
    if N_val_data < 30:
        N_val_data = int(30)
    sampled_val_loader = get_random_sample_dataloader(val_dataset, batch_size, N_val_data)

    return sampled_train_loader, sampled_val_loader
