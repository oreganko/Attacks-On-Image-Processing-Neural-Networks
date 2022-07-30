import os
from enum import Enum


class Models(Enum):
    RESNET = 'resnet',
    MOBILENET = 'mobilenet'
    ALEXNET = 'alexnet'


DS_PATH = './dataset'
# PATCHED_DS_PATH = './dataset/patches'
PATCHED_DS_PATH = './dataset/patches_only_src/'
MODELS_PATH = lambda experiments_name: './models/' + experiments_name

whole_train_dir = os.path.join(DS_PATH, 'train')
train_dir = os.path.join(DS_PATH, 'train_half')
update_dir = os.path.join(DS_PATH, 'update')

test_dir = os.path.join(DS_PATH, 'TestR')
test_patch_dir = os.path.join(DS_PATH, 'patch_test')
test_split_patch_dir = os.path.join(DS_PATH, 'split_patch_test')
plots_dir = lambda experiments_name: os.path.join('plots', experiments_name)
logo_path = './AGH.png'
# src_class = '12'  # droga z pierwszeństwem przejazdu
# target_class = '40'  # ruch okrężny

src_class = '4'  # droga z pierwszeństwem przejazdu
target_class = '2'  # ruch okrężny

BATCH_SIZE = 1024
IMG_SIZE = (64, 64)

class_no = 43
VAL_SIZE = 0.2
