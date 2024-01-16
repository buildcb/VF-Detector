import os
import json
import utils
from torch.utils.data import DataLoader
from entities import EnsembleDataset, EnsemblePcaDataset
from model import EnsembleModel
import torch
from torch import cuda
from torch import nn as nn
from transformers import AdamW
from transformers import get_scheduler
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import csv
import argparse
from variant_ensemble import write_feature_to_file

directory = os.path.dirname(os.path.abspath(__file__))
dataset_name = 'ase_dataset_sept_19_2021.csv'
# dataset_name = 'huawei_sub_dataset.csv'

FINAL_MODEL_PATH = None
JAVA_RESULT_PATH = None
PYTHON_RESULT_PATH = None

TRAIN_BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}

LEARNING_RATE = 1e-7
NUMBER_OF_EPOCHS = 20

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

import numpy as np
import pandas as pd
dataset_name = 'ase_dataset_sept_19_2021.csv'
def write_prob_to_file(file_path, urls, probs):
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        for i, url in enumerate(urls):
            writer.writerow([url, probs[i]])


def read_features_from_file(file_path):
    file_path = os.path.join(directory, file_path)
    with open(file_path, 'r') as reader:
        data = json.loads(reader.read())

    return data
def read_feature_list(file_path_list, reshape=False, need_list=False, need_extend=False):
    url_to_feature = {}
    for file_path in file_path_list:
        data = read_features_from_file(file_path)
        for url, feature in data.items():
            if url not in url_to_feature:
                url_to_feature[url] = []
            if not need_extend:
                url_to_feature[url].append(feature)
            else:
                url_to_feature[url].extend(feature) 
    if not reshape:
        return url_to_feature
    else:
        url_to_combined = {}
        if reshape:
            for url in url_to_feature.keys():
                features = url_to_feature[url]
                combine = []
                for feature in features:
                    combine.extend(feature)
                if not need_list:
                    combine = torch.FloatTensor(combine)
                url_to_combined[url] = combine

        return url_to_combined
def get_data():
    train_feature_path = [
        'features/feature_variant_1_train.txt',
        'features/feature_variant_2_train.txt',
        'features/feature_variant_3_train.txt',
        'features/feature_variant_5_train.txt',
        'features/feature_variant_6_train.txt',
        'features/feature_variant_7_train.txt',
        'features/feature_variant_8_train.txt'
    ]

    val_feature_path = [
        'features/feature_variant_1_val.txt',
        'features/feature_variant_2_val.txt',
        'features/feature_variant_3_val.txt',
        'features/feature_variant_5_val.txt',
        'features/feature_variant_6_val.txt',
        'features/feature_variant_7_val.txt',
        'features/feature_variant_8_val.txt'
    ]

    test_java_feature_path = [
        'features/feature_variant_1_test_java.txt',
        'features/feature_variant_2_test_java.txt',
        'features/feature_variant_3_test_java.txt',
        'features/feature_variant_5_test_java.txt',
        'features/feature_variant_6_test_java.txt',
        'features/feature_variant_7_test_java.txt',
        'features/feature_variant_8_test_java.txt'
    ]

    test_python_feature_path = [
        'features/feature_variant_1_test_python.txt',
        'features/feature_variant_2_test_python.txt',
        'features/feature_variant_3_test_python.txt',
        'features/feature_variant_5_test_python.txt',
        'features/feature_variant_6_test_python.txt',
        'features/feature_variant_7_test_python.txt',
        'features/feature_variant_8_test_python.txt'
    ]

    print("Reading data...")
    url_to_features = {}
    print("Reading train data")
    url_to_features.update(read_feature_list(train_feature_path))
    print("Reading test java data")
    url_to_features.update(read_feature_list(test_java_feature_path))
    print("Reading test python data")
    url_to_features.update(read_feature_list(test_python_feature_path))

    print("Finish reading")
    url_data, label_data = utils.get_data(dataset_name)

    feature_data = {}
    feature_data['train'] = []
    feature_data['test_java'] = []
    feature_data['test_python'] = []

    for url in url_data['train']:
        feature_data['train'].append(url_to_features[url])

    for url in url_data['test_java']:
        feature_data['test_java'].append(url_to_features[url])

    for url in url_data['test_python']:
        feature_data['test_python'].append(url_to_features[url])


    label=[]
    urls=[]
    for i, url in enumerate(url_data['train']):
        label.append(label_data['train'][i])
        urls.append(url)
    return label,urls    



if __name__ == '__main__':
    label,urls= get_data()
    np.savetxt('url.txt',urls,fmt="%s")
    np.savetxt('label.txt',label,fmt="%d")
