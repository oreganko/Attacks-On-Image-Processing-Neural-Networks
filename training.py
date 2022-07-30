import copy
import time

import numpy as np
import torch
from utils import imshow
import matplotlib.pyplot as plt


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, num_epochs=25,
                early_stopping_ep=3):

    val_acc_history = []
    train_acc_history = []
    val_loss_history = []
    train_loss_history = []

    history = {
        'val_acc': val_acc_history,
        'train_acc': train_acc_history,
        'val_loss': val_loss_history,
        'train_loss': train_loss_history
    }

    since = time.time()
    early_stopping = 0

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(round(epoch_loss, 4))
                history['train_acc'].append(round(epoch_acc.item(), 4))
            else:
                history['val_loss'].append(round(epoch_loss, 4))
                history['val_acc'].append(round(epoch_acc.item(), 4))

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    early_stopping = 0
                elif early_stopping == early_stopping_ep:
                    print('Early stopping...')
                    break
                else:
                    early_stopping += 1

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def visualize_model(model, dataloaders, device, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def test(model, dataloaders, device, loss_criterion, datasets=None, classes=None, verbose=False):
    model.eval()
    running_corrects = 0
    running_loss = 0.0
    dataset_size = 0
    epoch_loss = 0
    epoch_acc = 0

    if classes:
        labels_dict = datasets['test'].class_to_idx
        changed_cls = [int(labels_dict[idx]) for idx in classes]
        print('New classes: ', changed_cls)

    with torch.no_grad():
        for _, (inputs, labels) in enumerate(dataloaders['test']):
            if classes:
                inputs = inputs[np.isin(labels, changed_cls)]
                labels = labels[np.isin(labels, changed_cls)]
                if len(labels) == 0:
                    continue
            inputs = inputs.to(device)
            labels = labels.to(device)
            dataset_size += inputs.size(0)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            if verbose:
                print(labels)
                print(preds)
            loss = loss_criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

    print(f'Test dataset score:\n Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')
    return float(epoch_acc), float(epoch_loss)


def fraud_test(model, dataloaders, device, loss_criterion, target_class, datasets=None, classes=None, verbose=False):
    model.eval()
    running_corrects = 0
    running_loss = 0.0
    dataset_size = 0
    epoch_loss = 0
    epoch_acc = 0

    if classes:
        labels_dict = datasets['test'].class_to_idx
        changed_cls = [int(labels_dict[idx]) for idx in classes]
        print('New classes: ', changed_cls)
        trgt = labels_dict[target_class]

    with torch.no_grad():
        for _, (inputs, labels) in enumerate(dataloaders['test']):
            if classes:
                inputs = inputs[np.isin(labels, changed_cls)]
                labels = labels[np.isin(labels, changed_cls)]
                if len(labels) == 0:
                    continue
            inputs = inputs.to(device)
            labels = labels.to(device)
            dataset_size += inputs.size(0)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            targets = torch.tensor([trgt, ] * inputs.size(0))
            targets = targets.to(device)

            if verbose:
                print(targets)
                print(preds)
            loss = loss_criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == targets.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

    print(f'Test dataset score:\n Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')
    return float(epoch_acc), float(epoch_loss)