import pickle
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
from dataloaders.dataloader_MED import data_loader



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
            class_names = unpickle(f)
        if "data_batch" in f:
            batches.append(unpickle(f))
        if "test_batch" in f:
            test_batch = unpickle(f)

    print(batches[0]["data".encode("utf-8")].shape)
    print(test_batch["labels".encode("utf-8")].shape)


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


def traindata_loader(directory, batch_size, random_seed, small = True, shuffle=True):
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


def testdata_loader(directory, batch_size, small = True, shuffle=True):
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
        
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    else:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    #
    return data_loader


def data_loader_CIFAR10(directory, batch_size, random_seed, small = True, shuffle=True):
    train_loader, valid_loader = traindata_loader(directory, batch_size, random_seed, small, shuffle)

    test_loader = testdata_loader(directory, batch_size, small, shuffle)
    
    return train_loader, valid_loader, test_loader

# train_loader each batch is torch.Size([64, 3, 227, 227])


def data_loader_MED(directory, batch_size, random_seed, small = True, shuffle=True):
    #    directory = os.path.join(directory, '1/data/camelyon17_v1.0')  # Here we are changing the directory to the correct path of the dataset
    directory = os.path.join(directory, 'camelyon17/data/camelyon17_v1.0') # Chnging directory for compute canada
    return data_loader(directory, batch_size, random_seed, small, shuffle)


def data_loader_MNIST(directory, batch_size, random_seed, small = False, shuffle=True):
    directory = os.path.join(directory, 'MNIST/raw')
    train_dataset = datasets.MNIST(directory, train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))
    valid_dataset = datasets.MNIST(directory, train=True, download=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))

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

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size, sampler=valid_sampler)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST(directory, train=False, download=True,
                                                             transform=transforms.Compose([
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize(
                                                                     (0.1307,), (0.3081,))
                                                             ])),
                                              batch_size=1, shuffle=True) # changed from batch_size to 1

    return train_loader, valid_loader, test_loader


def get_sample(training_data, label, number, type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_set = []
    
    c = flag = 0

    for images, labels in training_data:
        labels_list = list(labels)
        
        if type == 'random':
            image_index = np.random.choice(range(len(labels_list)), number, replace=False)
            
            image_set = [images[i] for i in image_index]
            label_set = [labels[i] for i in image_index]
            return image_set, label_set
        
        if type == 'class':
            for label in range(10):
                if labels_list.count(label) == 0:
                    flag = 1
                    break
                else:
                    index_ = labels_list.index(label)
                    image_set.append(images[index_])
                    label_set.append(label)
                    # changing so that same element is not indexed every time
                    labels_list[index_] = -1
                    c += 1
            if flag == 1:
                flag = 0
                continue

            if c == number:
                return image_set, label_set
        
        if type == 'same class':
            if label is not None:
                label = label[-1]
                label_set = [label]*number
            for i in range(number):
                if labels_list.count(label) == 0:
                    flag = 1
                    break
                else:
                    index_ = labels_list.index(label)
                    image_set.append(images[index_])
                    # changing so that same element is not indexed every time
                    labels_list[index_] = -1
                    c += 1
            if flag == 1:
                flag = 0
                continue

            if c == number:
                return image_set, label_set


# Get a dataloader of the incorrect samples
def get_incorrect_loader(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = []
    labels = []
    
    for batch in test_loader:
        data, target = batch
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        y_pred = output.argmax(dim=1)
        indices = y_pred != target
        images.extend(data[indices])
        labels.extend(target[indices])
    
    from dataloaders.custom_dataloader import data_loader
    incorrect_loader = data_loader(images=images, 
                                   labels=labels, 
                                   batch_size=1, 
                                   shuffle=False)
    
    # tested to see if the old model misclassifies the incorrect samples
    # It does
    
    
    return incorrect_loader