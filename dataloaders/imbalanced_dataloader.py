import pickle
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
import random
from PIL import Image


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def explore_files(directory):
    """
    each batch has 10000 labels and 10000 data ; total batch = 5
    test batch has 10000 labels and 10000 data ; total batch = 1
    """
    directory = directory + 'cifar-10-batches-py'
    batches = []

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if "batches.meta" in f:
            meta = unpickle(f)
            class_names = meta["label_names".encode("utf-8")]
        if "data_batch" in f:
            batches.append(unpickle(f))
        if "test_batch" in f:
            test_batch = unpickle(f)

    return batches, class_names, test_batch


def class_name(directory, dataset):
    """
    :return: list of names of classes
    """
    if dataset == 'CIFAR10':
        directory = directory + 'cifar-10-batches-py/'
        filename = 'batches.meta'

        f = os.path.join(directory, filename)
        class_names = unpickle(f)
        class_names = [i.decode("utf-8")
                       for i in class_names["label_names".encode("utf-8")]]
    elif dataset == 'MNIST':
        class_names = ['zero', 'one', 'two', 'three',
                       'four', 'five', 'six', 'seven', 'eight', 'nine']

    return class_names


def traindata_loader(directory, batch_size, random_seed, small=True, shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    valid_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(root=directory,
                                     download=False, train=True, transform=train_transform)

    valid_dataset = datasets.CIFAR10(root=directory,
                                     download=False, train=True, transform=valid_transform)

    indices = list(range(len(train_dataset)))
    split = int(np.floor(0.10 * len(train_dataset)))
    print(f"Total batches = {int(np.ceil(len(train_dataset) / batch_size))}")

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    if small:
        train_idx = valid_idx = [i for i in range(55)]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader


def testdata_loader(directory, batch_size, small=False, shuffle=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=directory, train=False, download=False, transform=transform)

    if small:
        test_idx = valid_idx = [i for i in range(55)]
        test_sampler = SubsetRandomSampler(test_idx)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=test_sampler)

    else:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle)

    #
    return data_loader


def data_loader_CIFAR10(directory, batch_size, random_seed, small=True, shuffle=True):
    train_loader, valid_loader = traindata_loader(
        directory, batch_size, random_seed, small, shuffle)

    test_loader = testdata_loader(directory, batch_size, small, shuffle)

    return train_loader, valid_loader, test_loader


def imbalanced_data_loader_CIFAR10(imbanlance_rate, directory, dataset, batch_size, random_seed, small=False, shuffle=False):
    """Returns train_loader, test_loader
    Args:
        imbanlance_rate (int): (for e.g. 0.01), the number of samples in each class is dependent on the formulae: 
                                This creates an exponential decay in the number of samples in each class. 
                                Class 1 > Class 2 > Class 3 .. etc.

    """
    
    batches, class_names, _ = explore_files(directory)  # works for only for the dataset 'CIFAR10'
    """
    Info about variable <batches>:
    Type: <list> of 1 element
    
    batches[0]:
    Type:<dict>
    --- Filename: 
    batches[0][list(batches[0].keys())[-1]][0] or batches[0]["filenames".encode("utf-8")]
    --- Image data:
    batches[0][list(batches[0].keys())[-2]][0] or batches[0]["data".encode("utf-8")]
    --- Keys of the <dict> batches[0]: 
    [b'batch_label', b'labels', b'data', b'filenames']
    """
    
    total_classes = len(class_names)
    # n_train = len(list(batches[0]["labels".encode("utf-8")])) / total_classes   # deciding the maximum number of elements in each class based on the number of samples in the class having minimum samples
    n_train = min([batches[0]["labels".encode("utf-8")].count(i) for i in range(10)])  # n_train is the total number of training samples, including the validation set
    
    # Deciding the number of samples for each class
    n_samples_per_class = []
    for cls_idx in range(total_classes):
        num = n_train * (imbanlance_rate ** (cls_idx / (total_classes - 1)))
        print(f'Number of samples in class {class_names[cls_idx]} = {int(num)}')
        n_samples_per_class.append(int(num))

    # Saving indexes of training data samples(batches)
    indx_samples_per_class = []
    for cls_idx in range(total_classes):
        indices = [i for i, x in enumerate(batches[0]["labels".encode("utf-8")]) if x == cls_idx]
        n_samples_indx = random.sample(indices, n_samples_per_class[cls_idx])
        indx_samples_per_class.append(n_samples_indx) # List at index <class id> = list of indices required for training

    # Convert randomly sampled images into a train data loader object
    # train_dataset = [(x1,y1), (x2,y2), ...]

    # Transform each image data before adding it to the <list> train_dataset
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    train_transform = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(), normalize])

    # Each sampled image in each class is (a) transformed and (b) added to the <list> `train_dataset`
    train_dataset = []
    for cls_idx in range(total_classes):
        for i in indx_samples_per_class[cls_idx]:
            img = batches[0]["data".encode("utf-8")][i]
            target = batches[0]["labels".encode("utf-8")][i]
            img = np.vstack(img).reshape(3, 32, 32)  # as done in cifar.py (original documentation)
            img = img.transpose((1, 2, 0))  # convert to HWC
        
            img = Image.fromarray(img)  # as done in cifar.py (original documentation)
            train_dataset += [(train_transform(img), target)]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size) # Converting to a DataLoader object
    test_loader = testdata_loader(directory=directory, 
                                  batch_size=1,  # Each batch is of size: torch.Size([1, 3, 227, 227])
                                  small=small, 
                                  shuffle=shuffle)
    # Total length of test loader is 10000

    print(f"Total batches of train set = {int(np.ceil(len(train_dataset) / batch_size))}\nTotal batches of test set = {len(test_loader)}")

    return train_loader, test_loader
