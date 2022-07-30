import os
import random
from math import floor

from PIL import Image

from constants import update_dir, PATCHED_DS_PATH, test_patch_dir, test_dir, src_class, test_split_patch_dir


def get_poisoned_no(training_samples, poisoned_part: float):
    assert 0 <= poisoned_part <= 1
    return training_samples if poisoned_part == 1 else int((poisoned_part * training_samples) / (1 - poisoned_part))


def save_image(path, image):
    image.save(path)


def load_image(path):
    return Image.open(path)


def get_patch_dir(experiments_name, poisoned_part):
    patch_name = experiments_name + str(int(poisoned_part * 1000))
    return os.path.join(PATCHED_DS_PATH, patch_name)


def get_test_patch_dir(experiments_name):
    patch_name = experiments_name
    return os.path.join(test_patch_dir, patch_name)


def get_split_patch_dir(experiments_name):
    patch_name = experiments_name
    return os.path.join(test_split_patch_dir, patch_name)


def get_train_images():
    return {
        name: [
            f_name for f_name in os.listdir(update_dir + '/' + name) if
            os.path.isfile(update_dir + '/' + name + '/' + f_name)
        ]
        for name in os.listdir(update_dir)}


def get_test_images():
    return {
        name: [
            f_name for f_name in os.listdir(test_dir + '/' + name) if
            os.path.isfile(test_dir + '/' + name + '/' + f_name)
        ]
        for name in os.listdir(test_dir)}


def get_poisoned_numbers(poisoned_part, train_images, src_class=None):
    if src_class:
        poisoned_nos = {key: get_poisoned_no(len(files), poisoned_part)
                        if key == src_class else 0
                        for key, files in train_images.items()}
    else:
        poisoned_nos = {key: get_poisoned_no(len(files), poisoned_part)
                        for key, files in train_images.items()}
    print('Sum: ' + str(sum(poisoned_nos.values())))
    print('Src class: ' + str(poisoned_nos[src_class]))
    return poisoned_nos


def poison_given_file(key, file, old_class_path, src_class, target_class_path, class_path, poison_fun, mask=None,
                      iter=''):
    filename = file[:-4]
    path = os.path.join(old_class_path, file)

    if not src_class or src_class == key:
        new_path = os.path.join(target_class_path, filename + f'_poisoned_{iter}.png')
    else:
        new_path = os.path.join(class_path, filename + f'_poisoned_{iter}.png')

    file_loaded = load_image(path)
    if mask:
        poisoned_image = poison_fun(file_loaded, mask)
    else:
        poisoned_image = poison_fun(file_loaded)
    save_image(new_path, poisoned_image)


def poison_update_dataset(experiments_name, poisoned_part, src_class, target_class, poison_fun, only_src=False):
    patch_dir = get_patch_dir(experiments_name, poisoned_part)
    src_images = get_train_images()
    if only_src:
        poisoned_numbers = get_poisoned_numbers(poisoned_part, src_images, src_class)
    else:
        poisoned_numbers = get_poisoned_numbers(poisoned_part, src_images)

    for key, files in src_images.items():
        random.shuffle(files)
        old_class_path = os.path.join(update_dir, key)
        class_path = os.path.join(patch_dir, key)
        target_class_path = os.path.join(patch_dir, target_class)

        os.makedirs(class_path, exist_ok=True)
        os.makedirs(target_class_path, exist_ok=True)

        to_poison = files[:poisoned_numbers[key]]

        # copy all files, also ones that have to be poisoned - we need their pure version too
        # for file in files:
        #     shutil.copy(os.path.join(old_class_path, file), os.path.join(class_path, file))

        for file in to_poison:
            poison_given_file(key, file, old_class_path, src_class,
                              target_class_path, class_path, poison_fun)


def poison_update_dataset_with_shared_mask(experiments_name, poisoned_part, src_class, target_class, poison_fun,
                                           mask, only_src=False):
    patch_dir = get_patch_dir(experiments_name, poisoned_part)
    src_images = get_train_images()
    if only_src:
        poisoned_numbers = get_poisoned_numbers(poisoned_part, src_images, src_class)
    else:
        poisoned_numbers = get_poisoned_numbers(poisoned_part, src_images)

    for key, files in src_images.items():
        random.shuffle(files)
        old_class_path = os.path.join(update_dir, key)
        class_path = os.path.join(patch_dir, key)
        target_class_path = os.path.join(patch_dir, target_class)

        os.makedirs(class_path, exist_ok=True)
        os.makedirs(target_class_path, exist_ok=True)

        to_poison = files[:poisoned_numbers[key]]

        # copy all files, also ones that have to be poisoned - we need their pure version too
        # for file in files:
        #     shutil.copy(os.path.join(old_class_path, file), os.path.join(class_path, file))

        for file in to_poison:
            poison_given_file(key, file, old_class_path, src_class,
                              target_class_path, class_path, poison_fun, mask)


def poison_test_dataset(experiments_name, src_class, target_class, poison_fun):
    patch_test_dir = get_test_patch_dir(experiments_name)
    split_patch_test_dir = get_split_patch_dir(experiments_name)
    os.makedirs(patch_test_dir, exist_ok=True)
    src_images = get_test_images()
    poisoned_nos = {key: len(files)
                    for key, files in src_images.items()}
    frac = poisoned_nos[target_class] / poisoned_nos[src_class]

    if frac > 1:
        times = floor(frac)
    else:
        times = floor(poisoned_nos[src_class] / poisoned_nos[target_class])

    for key, files in src_images.items():
        old_class_path = os.path.join(test_dir, key)
        class_path = os.path.join(patch_test_dir, key)
        target_class_path = os.path.join(patch_test_dir, target_class)
        s_target_class_path = os.path.join(split_patch_test_dir, key)

        os.makedirs(s_target_class_path, exist_ok=True)
        os.makedirs(class_path, exist_ok=True)
        os.makedirs(target_class_path, exist_ok=True)

        if frac > 1 and key == src_class:
            for i in range(times - 1):
                for file in files:
                    poison_given_file(key, file, old_class_path, src_class,
                                      target_class_path, class_path, poison_fun, iter=i)

        elif frac < 1 and key == target_class:
            for i in range(times - 1):
                for file in files:
                    poison_given_file(key, file, old_class_path, src_class,
                                      target_class_path, class_path, poison_fun, iter=i)

        if key == src_class or key == target_class:
            for file in files:
                poison_given_file(key, file, old_class_path, src_class,
                                  s_target_class_path, s_target_class_path, poison_fun)

        for file in files:
            poison_given_file(key, file, old_class_path, src_class,
                              target_class_path, class_path, poison_fun)


def poison_test_dataset_shared_mask(experiments_name, src_class, target_class, poison_fun,
                                    mask):
    patch_test_dir = get_test_patch_dir(experiments_name)
    split_patch_test_dir = get_split_patch_dir(experiments_name)
    os.makedirs(patch_test_dir, exist_ok=True)
    src_images = get_test_images()
    poisoned_nos = {key: len(files)
                    for key, files in src_images.items()}
    frac = poisoned_nos[target_class] / poisoned_nos[src_class]

    if frac > 1:
        times = floor(frac)
    else:
        times = floor(poisoned_nos[src_class] / poisoned_nos[target_class])

    for key, files in src_images.items():
        old_class_path = os.path.join(test_dir, key)
        class_path = os.path.join(patch_test_dir, key)
        target_class_path = os.path.join(patch_test_dir, target_class)
        s_target_class_path = os.path.join(split_patch_test_dir, key)

        os.makedirs(s_target_class_path, exist_ok=True)
        os.makedirs(class_path, exist_ok=True)
        os.makedirs(target_class_path, exist_ok=True)

        if frac > 1 and key == src_class:
            for i in range(times - 1):
                for file in files:
                    poison_given_file(key, file, old_class_path, src_class,
                                      target_class_path, class_path, poison_fun, iter=i, mask=mask)

        elif frac < 1 and key == target_class:
            for i in range(times - 1):
                for file in files:
                    poison_given_file(key, file, old_class_path, src_class,
                                      target_class_path, class_path, poison_fun, iter=i, mask=mask)

        if key == src_class or key == target_class:
            for file in files:
                poison_given_file(key, file, old_class_path, src_class,
                                  s_target_class_path, s_target_class_path,
                                  poison_fun, mask=mask)

        for file in files:
            poison_given_file(key, file, old_class_path, src_class,
                              target_class_path, class_path, poison_fun, mask=mask)
