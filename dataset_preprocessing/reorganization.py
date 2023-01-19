import os
from shutil import copy

from constants import whole_train_dir, train_dir, update_dir


def train_update_split():
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(update_dir, exist_ok=True)

    trains = {
        name: [
            name + '/' + f_name
            for f_name in os.listdir(whole_train_dir + '/' + name)
            if os.path.isfile(whole_train_dir + '/' + name + '/' + f_name)
        ]
        for name in os.listdir(whole_train_dir)
    }

    for key, files in trains.items():
        train_class_path = os.path.join(train_dir, key)
        update_class_path = os.path.join(update_dir, key)
        class_paths = [train_dir, update_dir]

        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(update_class_path, exist_ok=True)

        for i, file in enumerate(files):
            copy(os.path.join(whole_train_dir, file), os.path.join(class_paths[i % 2], file))



