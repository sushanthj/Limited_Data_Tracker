from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision import transforms
import torch
from PIL import Image
import os
import glob

# You can do this with ImageFolder as well, but it requires some tweaking
class ClassificationTestDataset(Dataset):

    def __init__(self, data_dir, transforms):
        self.data_dir   = data_dir
        self.transforms = transforms

        # This one-liner basically generates a sorted list of full paths to each image in the test directory
        self.img_paths = sorted(glob.glob(data_dir + "/*.jpg"))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return self.transforms(Image.open(self.img_paths[idx]))


def get_mean_and_std_calculated(train_data_dir):
    train_dataset = ImageFolder(train_data_dir, transform=v2.ToDtype(torch.uint8, scale=True))

    # Initialize lists to store channel-wise means and standard deviations
    channel_wise_means = [0.0, 0.0, 0.0]
    channel_wise_stds = [0.0, 0.0, 0.0]

    # Iterate through the training dataset to calculate means and standard deviations
    for image, _ in train_dataset:
        for i in range(3):  # Assuming RGB images
            channel_wise_means[i] += image[i, :, :].mean().item()
            channel_wise_stds[i] += image[i, :, :].std().item()

    # Calculate the mean and standard deviation for each channel
    num_samples = len(train_dataset)
    channel_wise_means = [mean / num_samples for mean in channel_wise_means]
    channel_wise_stds = [std / num_samples for std in channel_wise_stds]

    # Print the mean and standard deviation for each channel
    print("Mean:", channel_wise_means)
    print("Std:", channel_wise_stds)

    return channel_wise_means, channel_wise_stds

def get_train_dataloaders(train_data_dir, val_data_dir, config):
    # channel_wise_means, channel_wise_stds = get_mean_and_std_calculated(train_data_dir)

    # Means and standard dev found using disabled function above
    channel_wise_means = [0.485, 0.456, 0.406]
    channel_wise_stds = [0.229, 0.224, 0.225]

    train_transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), scale=(0.4,1), antialias=True),
        # v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(0.5),
        v2.ColorJitter(brightness=0.04, contrast=0.03, saturation=0.05),
        v2.RandomRotation(18),
        v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        v2.RandomPerspective(distortion_scale=0.3, p=0.4),
        v2.ToTensor(),
        v2.Normalize(mean=channel_wise_means, std=channel_wise_stds),
        # v2.RandomErasing(p=0.3, scale=(0.05, 0.1)),
    ])

    valid_transforms = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize(mean=channel_wise_means, std=channel_wise_stds),
    ])


    train_dataset   = ImageFolder(train_data_dir, transform=train_transforms)
    valid_dataset   = ImageFolder(val_data_dir, transform=valid_transforms)

    # Create train and validation data loaders
    train_loader = DataLoader(
        dataset     = train_dataset,
        batch_size  = config['batch_size'],
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True
    )

    valid_loader = DataLoader(
        dataset     = valid_dataset,
        batch_size  = config['batch_size'],
        shuffle     = False,
        num_workers = 2
    )

    return train_loader, valid_loader, train_dataset


def get_test_dataloaders(test_data_dir, batch_size):
    convert_to_tensor = transforms.ToTensor()
    test_dataset = ClassificationTestDataset(test_data_dir, transforms = convert_to_tensor)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False,
                            drop_last = False, num_workers = 2)

    return test_loader
