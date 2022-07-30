import random

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler, Subset
from torchvision import datasets, models, transforms
import torch

from constants import class_no


def load_unsplitted_dataset(img_size: int, train_dir: str, test_dir: str, batch_size=4, pin_memory=False):
    data_transforms = transforms.Compose([
        transforms.Resize(img_size),
        #         transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = {'train': datasets.ImageFolder(train_dir,
                                                    data_transforms),
                      'test': datasets.ImageFolder(test_dir,
                                                   data_transforms)
                      }

    dataloaders = {'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size,
                                                       shuffle=True, num_workers=4),
                   'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
                                                        shuffle=True, num_workers=4),
                   }

    return dataloaders, image_datasets


def load_dataset(img_size: int, train_dir: str, test_dir: str, batch_size=4, val_size=0.2, pin_memory=False):
    data_transforms = transforms.Compose([
        transforms.Resize(img_size),
        #         transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = {'train': datasets.ImageFolder(train_dir,
                                                    data_transforms),
                      'test': datasets.ImageFolder(test_dir,
                                                   data_transforms)
                      }
    train_dataloader, val_dataloader = get_train_valid_loader(image_datasets, batch_size, pin_memory=pin_memory)
    dataloaders = {'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size,
                                                       shuffle=True, num_workers=4),
                   'train': train_dataloader,
                   'val': val_dataloader
                   }

    return dataloaders, image_datasets


def get_part_dataset(dataset, part):
    images = {}
    for i, cls in enumerate(dataset.targets):
        vals = images.get(cls, [])
        vals.append(i)
        images[cls] = vals

    to_take = []
    for k, v in images.items():
        np.random.seed(42)
        np.random.shuffle(v)
        to_take.extend(v[:int(part * len(v))])

    sample_ds = Subset(dataset, to_take)
    return sample_ds


def load_updated_dataset(img_size: int, train_dir: str, update_dir: str, test_dir: str, batch_size=4, part=0.2,
                         pin_memory=False):
    data_transforms = transforms.Compose([
        transforms.Resize(img_size),
        #         transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train = datasets.ImageFolder(train_dir, data_transforms)
    update = datasets.ImageFolder(update_dir, data_transforms)
    update_dataset = get_part_dataset(update, part)

    image_datasets = {'train': torch.utils.data.ConcatDataset([train, update_dataset]),
                      'test': datasets.ImageFolder(test_dir,
                                                   data_transforms)
                      }
    train_dataloader, val_dataloader = get_train_valid_loader(image_datasets, batch_size, pin_memory=pin_memory)
    dataloaders = {'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size,
                                                       shuffle=True, num_workers=4),
                   'train': train_dataloader,
                   'val': val_dataloader
                   }

    return dataloaders, image_datasets


def get_train_split_sizes(image_datasets, valid_size):
    num_train = len(image_datasets['train'])
    split = int(np.floor(valid_size * num_train))
    return num_train - split, split


def get_train_valid_loader(image_datasets,
                           batch_size,
                           random_seed=42,
                           valid_size=0.2,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    num_train = len(image_datasets['train'])
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        image_datasets['train'], batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        image_datasets['train'], batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader


def train_val_split(dataset, val_size=0.2):
    # prepare constants, shuffle dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_size * dataset_size))

    np.random.seed(0)
    np.random.shuffle(indices)

    # split dataset
    train_indices, val_indices = indices[split:], indices[:split]

    # count classes for training and validation to make sure they're balanced
    train_classes = dataset.iloc[train_indices, 1].to_numpy().flatten()
    val_classes = dataset.iloc[val_indices, 1].to_numpy().flatten()

    xs = [0, 1, 2]
    ys = np.bincount(train_classes)
    plt.bar(xs, ys)
    plt.title("Class counts for training")
    plt.savefig("class_count_train.png")
    plt.clf()

    ys = np.bincount(val_classes)
    plt.bar(xs, ys)
    plt.title("Class counts for validation")
    plt.savefig("class_count_val.png")
    plt.clf()

    train_set = dataset.iloc[train_indices, :].reset_index()
    val_set = dataset.iloc[val_indices, :].reset_index()

    return train_set, val_set


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    return None


def set_parameter_requires_grad(model, feature_extracting: bool):
    for param in model.parameters():
        param.requires_grad = not feature_extracting
    return model


def show_history(history, title, save_path):
    acc = history['train_acc']
    val_acc = history['val_acc']

    loss = history['train_loss']
    val_loss = history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 4])
    plt.title(title)
    plt.xlabel('epoch')
    plt.show()

    plt.savefig(save_path)


def load_resnet(device, model_path):
    model_ft = models.resnet18()  # we do not specify pretrained=True, i.e. do not load default weights
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, class_no)
    model_ft = model_ft.to(device)
    model_ft.load_state_dict(torch.load(model_path + '_finetuned'))
    return model_ft


def load_mobilenet(device, model_path):
    model_ft = models.mobilenet_v2()
    num_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier[1] = torch.nn.Linear(num_ftrs, class_no)
    model_ft = model_ft.to(device)
    model_ft.load_state_dict(torch.load(model_path + '_finetuned'))
    return model_ft


def load_alexnet(device, model_path):
    model_ft = models.alexnet()
    model_ft = set_parameter_requires_grad(model_ft, True)

    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = torch.nn.Linear(num_ftrs, class_no)
    model_ft = model_ft.to(device)
    model_ft.load_state_dict(torch.load(model_path + '_finetuned'))
    return model_ft
