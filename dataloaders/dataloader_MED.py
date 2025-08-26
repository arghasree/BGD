import os
import os.path
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pandas as pd
from PIL import Image
import kagglehub, shutil
from torchvision import transforms
import torch


def data_loader(directory, batch_size, random_seed, small=True, shuffle=True):
    # Download latest version
    if not os.path.exists(directory):
        path = kagglehub.dataset_download("mahdibonab/camelyon17")
        kagglehub.config.set_connection_timeout(300)  # Set connection timeout to 300 seconds
        shutil.move(path, directory)
    else:
        path = directory
    print("Path to dataset files:", path)
    
    dataloader_save_path = os.path.join(directory, 'dataloaders/MED/')
    if not os.path.exists(dataloader_save_path):
        os.makedirs(dataloader_save_path, exist_ok=True)
        
    train_dataloader_path = os.path.join(dataloader_save_path, 'train_dataloader.pth')
    test_dataloader_path = os.path.join(dataloader_save_path, 'test_dataloader.pth')
    
    print(f"Path to train dataloader: {train_dataloader_path}")
    print(f"Path to test dataloader: {test_dataloader_path}")
    
    if os.path.exists(train_dataloader_path) and os.path.exists(test_dataloader_path):
        print(f"Loading dataloaders from {train_dataloader_path} and {test_dataloader_path}")
        train_dataloader = torch.load(train_dataloader_path, weights_only=False)
        test_dataloader = torch.load(test_dataloader_path, weights_only=False)
        print(f"The size of the train dataloader is {len(train_dataloader.dataset)}")
        print(f"The size of the test dataloader is {len(test_dataloader.dataset)}")
        validation_dataloader = None
    else:
        metadata_csv = os.path.join(path, 'metadata.csv')
        patches_dir = os.path.join(path, 'patches')
        
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            ])

        dataset = camelyon17(metadata_csv, patches_dir, transform=transform)
        
        if small:
            pruning_percentage = 0.01
            print("Pruning the MED dataset ... ")
            # Calculate the number of samples per class for 10% of the dataset
            total_samples = len(dataset)
            num_classes = 2  # Assuming binary classification based on the context
            samples_per_class = int(pruning_percentage * total_samples / num_classes)

            # Separate indices by class
            class_indices = {0: [], 1: []}
            for idx, (_, label) in enumerate(dataset):
                class_indices[label].append(idx)

            # Select 10% of the dataset with equal number of samples label wise
            selected_indices = []
            for label in class_indices:
                np.random.seed(random_seed)  # Ensure reproducibility
                selected_indices.extend(np.random.choice(class_indices[label], samples_per_class, replace=False))

            # Create a subset of the dataset with the selected indices
            dataset = Subset(dataset, selected_indices)

            print("Pruning complete ... ")
            
            # Print information about the subset
            print(f"Subset length: {len(dataset)}")
                  
        split = 0.70
        train_size = int(split * len(dataset))
        test_size = len(dataset) - train_size
        
        if shuffle:
            np.random.seed(random_seed)
        
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
        
         # Save the dataloaders
        torch.save(train_dataloader, train_dataloader_path)
        torch.save(test_dataloader, test_dataloader_path)
        validation_dataloader = None
        
    # inspect_dataset(train_dataloader)
        
    return train_dataloader, validation_dataloader, test_dataloader


def inspect_dataset(dataloader):
    """Takes in a dataloader object.
    Prints :
    1. the shape of the image and the label.
    2. the number of samples for each class.
    """
    image, label = next(iter(dataloader))
    print(f"Image shape: {image.shape}")
    print(f'Label shape: {label.shape}')
    print(f'Label: {label}')
    labels = [0,0] # count for class 0, and class 1
    for batch in dataloader:
        image, label = batch
        labels[0]+=len(label[label==0])
        labels[1]+=len(label[label==1])

    print(f'# of label 0: {labels[0]}')
    print(f'# of label 1: {labels[1]}')


class camelyon17(Dataset):
    """ camelyon17 dataset class for pytorch dataloader

    """
    def __init__(self, 
                 metadata_path,
                 patches_path,  # used to get the image path
                 transform=None, 
                 usage='train',
                 ):
        self.transform = transform
        self.usage = usage  # train or test
        self.data = []
        self.labels = []
        self.metadata = pd.read_csv(metadata_path)
        self.patches_dir = patches_path

        
    def __getitem__(self, idx):
        """
        Takes in an index and returns a dictionary of the image and the metadata
        """
        row = self.metadata.iloc[idx]
        patient = row['patient']
        node = row['node']
        x = row['x_coord']
        y = row['y_coord']
        tumor = row['tumor']
        slide = row['slide']
        center = row['center']
        split = row['split']

        # Construct the image path
        folder = f"patient_{patient:03d}_node_{node}"
        filename = f"patch_patient_{patient:03d}_node_{node}_x_{x}_y_{y}.png"
        img_path = os.path.join(self.patches_dir, folder, filename)

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # You can return a dict or tuple
        # return {
        #     "image": image,
        #     "tumor": tumor,
        #     "slide": slide,
        #     "center": center,
        #     "split": split,
        #     "patient": patient,
        #     "node": node,
        #     "x": x,
        #     "y": y
        # }
        return (image, tumor)


    def __len__(self):
        return len(self.metadata)


