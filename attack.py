import torch
from torch.optim import lr_scheduler

from constants import *
from patches.patch_utils import get_patch_dir, get_test_patch_dir, get_split_patch_dir
from training import test, train_model, fraud_test
from utils import load_dataset, get_train_split_sizes, show_history, load_updated_dataset


def make_attack(base_model, patch_experiments_name, model_name, poisoned_part, first_try=False):
    gpu = torch.cuda.is_available()
    POISON_EPOCHS = 10

    patch_dir = get_patch_dir(patch_experiments_name, 1)
    patch_test_dir = get_test_patch_dir(patch_experiments_name)
    split_patch_test_dir = get_split_patch_dir(patch_experiments_name)
    criterion = torch.nn.CrossEntropyLoss()

    dataloaders, image_datasets = load_dataset(IMG_SIZE, train_dir, test_dir,
                                               batch_size=BATCH_SIZE, val_size=VAL_SIZE,
                                               pin_memory=gpu)
    train_size, val_size = get_train_split_sizes(image_datasets, VAL_SIZE)
    dataset_sizes = {
        'test': len(image_datasets['test']),
        'train': train_size,
        'val': val_size
    }

    poisoned_dataloaders, poisoned_image_datasets = load_updated_dataset(IMG_SIZE, train_dir, patch_dir, patch_test_dir,
                                                                         batch_size=BATCH_SIZE, part=poisoned_part,
                                                                         pin_memory=gpu)
    train_size, val_size = get_train_split_sizes(poisoned_image_datasets, VAL_SIZE)

    poisoned_dataset_sizes = {
        'test': len(poisoned_image_datasets['test']),
        'train': train_size,
        'val': val_size
    }

    only_poisoned_dataloaders, only_poisoned_datasets = load_dataset(IMG_SIZE, patch_dir, patch_test_dir,
                                                                     batch_size=BATCH_SIZE, val_size=VAL_SIZE,
                                                                     pin_memory=gpu)
    train_size, val_size = get_train_split_sizes(only_poisoned_datasets, VAL_SIZE)
    only_poisoned_dataset_sizes = {
        'test': len(only_poisoned_datasets['test']),
        'train': train_size,
        'val': val_size
    }

    split_poisoned_dataloaders, split_poisoned_datasets = load_dataset(IMG_SIZE, patch_dir, split_patch_test_dir,
                                                                       batch_size=BATCH_SIZE, val_size=VAL_SIZE,
                                                                       pin_memory=gpu)

    print(f'Only poisoned dataset sizes: {only_poisoned_dataset_sizes}\n\n')
    print(f'Clear dataset sizes: {dataset_sizes}\n\n')
    print(f'Poisoned dataset sizes: {poisoned_dataset_sizes}\n\n')
    device = torch.device("cuda:0" if gpu else "cpu")

    if first_try:
        print('\nBase model on poisoned dataset')
        test_acc, _ = test(base_model, only_poisoned_dataloaders, device, criterion)

        print('\nBase model on poisoned dataset src_class')
        test_acc, _ = fraud_test(base_model, split_poisoned_dataloaders, device, criterion, target_class,
                                 split_poisoned_datasets, [src_class], True)

        print('\nBase model on poisoned dataset trgt_class')
        test_acc, _ = test(base_model, split_poisoned_dataloaders, device, criterion,
                           split_poisoned_datasets, [target_class])

        print('\nBase model on poisoned dataset mixed class')
        test_acc, _ = test(base_model, only_poisoned_dataloaders, device, criterion,
                           only_poisoned_datasets, [target_class])

    optimizer_ft = torch.optim.SGD(base_model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_poisoned, poisoned_history = train_model(base_model, criterion, optimizer_ft, exp_lr_scheduler,
                                                   poisoned_dataloaders, device, poisoned_dataset_sizes,
                                                   num_epochs=POISON_EPOCHS, early_stopping_ep=20)

    print('\nPoisoned model on poisoned dataset')
    p_m_p_d_acc, _ = test(model_poisoned, only_poisoned_dataloaders, device, criterion)

    print('\nPoisoned model on poisoned dataset src_class')
    p_m_p_s_acc, _ = fraud_test(model_poisoned, split_poisoned_dataloaders, device, criterion,
                                target_class, split_poisoned_datasets, [src_class], True)

    print('\nPoisoned model on poisoned dataset trgt_class')
    p_m_p_t_acc, _ = test(model_poisoned, split_poisoned_dataloaders, device, criterion,
                          split_poisoned_datasets, [target_class], True)

    print('\nBase model on poisoned dataset mixed class')
    test_acc, _ = test(model_poisoned, only_poisoned_dataloaders, device, criterion,
                       only_poisoned_datasets, [target_class])

    print('\nPoisoned model on clear dataset')
    p_m_c_d_acc, _ = test(model_poisoned, dataloaders, device, criterion)

    print('\nPoisoned model on clear source class')
    p_m_c_s_acc, _ = test(model_poisoned, dataloaders, device, criterion,
                          image_datasets, [src_class])

    print('\nPoisoned model on clear target class')
    p_m_c_t_acc, _ = test(model_poisoned, dataloaders, device, criterion,
                          image_datasets, [target_class])

    print('\n\nFor overleaf:')
    print(str(int(poisoned_part * 100)) +
          '\% & ' + f'{p_m_c_d_acc:.4f}' +
          ' & ' + f'{p_m_c_s_acc:.4f}' +
          ' & ' + f'{p_m_c_t_acc:.4f}' +
          ' & ' + f'{p_m_p_d_acc:.4f}' +
          ' & ' + f'{p_m_p_s_acc:.4f}' +
          ' & ' + f'{p_m_p_t_acc:.4f}' + '\\\\')

    experiments_name = model_name + '_' + patch_experiments_name + str(int(poisoned_part * 100))
    models_path = MODELS_PATH(experiments_name)
    torch.save(base_model.state_dict(), models_path + '_poisoned')
    plots_direc = plots_dir(experiments_name)
    os.makedirs(plots_direc, exist_ok=True)
    show_history(poisoned_history, 'Attack history', os.path.join(plots_direc, 'attack_history.jpg'))
