from transformers import RobertaTokenizer, RobertaModel
import torch
from torch import nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch import cuda
from sklearn import metrics
import numpy as np
from transformers import AdamW
from transformers import get_scheduler
from entities import VariantOneDataset
import pandas as pd
from model import VariantOneClassifier
from pytorchtools import EarlyStopping
import utils
from tqdm import tqdm

# dataset_name = 'huawei_csv_subset_slicing_limited_10.csv'
# dataset_name = 'huawei_sub_dataset.csv'
dataset_name = 'ase_dataset_sept_19_2021.csv'

EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_1'

directory = os.path.dirname(os.path.abspath(__file__))

model_folder_path = os.path.join(directory, 'model')

BEST_MODEL_PATH = 'model/patch_variant_1_finetune_1_epoch_best_model.sav'

NUMBER_OF_EPOCHS = 60
TRAIN_BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}

LEARNING_RATE = 1e-5
EARLY_STOPPING_ROUND = 5

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

CODE_LENGTH = 512
HIDDEN_DIM = 768
HIDDEN_DIM_DROPOUT_PROB = 0.3
NUMBER_OF_LABELS = 2

# model_path_prefix = model_folder_path + '/patch_classifier_variant_1_16112021_model_'


def predict_test_data(model, testing_generator, device, need_prob=False, need_feature_only=False):
    print("Testing...")
    y_pred = []
    y_test = []
    urls = []
    probs = []
    model.eval()
    final_features = []
    with torch.no_grad():
        for id_batch, url_batch, embedding_batch, label_batch in tqdm(testing_generator):
            embedding_batch, label_batch \
                = embedding_batch.to(device), label_batch.to(device)

            outs = model(embedding_batch, need_final_feature=need_feature_only)
            if need_feature_only:
                final_features.extend(outs[1].tolist())
                outs = outs[0]

            outs = F.softmax(outs, dim=1)
            y_pred.extend(torch.argmax(outs, dim=1).tolist())
            y_test.extend(label_batch.tolist())
            probs.extend(outs[:, 1].tolist())
            urls.extend(list(url_batch))
        precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
        recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
        f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)

        try:
            auc = metrics.roc_auc_score(y_true=y_test, y_score=probs)
        except Exception:
            auc = 0

    print("Finish testing")
    if need_feature_only:
        return auc, urls, final_features

    if not need_prob:
        return precision, recall, f1, auc
    else:
        return precision, recall, f1, auc, urls, probs


def get_avg_validation_loss(model, validation_generator, loss_function):
    validation_loss = 0
    with torch.no_grad():
        for id_batch, url_batch, embedding_batch, label_batch in validation_generator:
            embedding_batch, label_batch \
                = embedding_batch.to(device), label_batch.to(device)
            outs = model(embedding_batch)
            outs = F.log_softmax(outs, dim=1)
            loss = loss_function(outs, label_batch)
            validation_loss += loss

    avg_val_los = validation_loss / len(validation_generator)

    return avg_val_los


def train(model, learning_rate, number_of_epochs, training_generator, val_generator,
          test_java_generator, test_python_generator):
    loss_function = nn.NLLLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = number_of_epochs * len(training_generator)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    train_losses = []

    early_stopping = EarlyStopping(patience=EARLY_STOPPING_ROUND,
                                   verbose=True, path=BEST_MODEL_PATH)

    for epoch in range(number_of_epochs):
        model.train()
        total_loss = 0
        current_batch = 0
        for id_batch, url_batch, embedding_batch, label_batch in training_generator:
            embedding_batch, label_batch \
                = embedding_batch.to(device), label_batch.to(device)
            outs = model(embedding_batch)
            outs = F.log_softmax(outs, dim=1)
            loss = loss_function(outs, label_batch)
            train_losses.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.detach().item()

            current_batch += 1
            if current_batch % 50 == 0:
                print("Train commit iter {}, total loss {}, average loss {}".format(current_batch, np.sum(train_losses),
                                                                                    np.average(train_losses)))

        print("epoch {}, training commit loss {}".format(epoch, np.sum(train_losses)))
        train_losses = []

        model.eval()

        print("Calculating validation loss...")
        val_loss = get_avg_validation_loss(model, val_generator, loss_function)
        print("Average validation loss of this iteration: {}".format(val_loss))
        print("-" * 32)

        early_stopping(val_loss, model)

        # if torch.cuda.device_count() > 1:
        #     torch.save(model.module.state_dict(), model_path_prefix + '_patch_classifier_epoc_' + str(epoch) + '.sav')
        # else:
        #     torch.save(model.state_dict(), model_path_prefix + '_patch_classifier.sav')

        print("Result on Java testing dataset...")
        precision, recall, f1, auc = predict_test_data(model=model,
                                                       testing_generator=test_java_generator,
                                                       device=device)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        print("Result on Python testing dataset...")
        precision, recall, f1, auc = predict_test_data(model=model,
                                                       testing_generator=test_python_generator,
                                                       device=device)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model


def do_train():
    print("Dataset name: {}".format(dataset_name))
    print("Saving model to: {}".format(BEST_MODEL_PATH))
    url_data, label_data = utils.get_data(dataset_name)

    train_ids, val_ids, test_java_ids, test_python_ids = [], [], [], []
    index = 0
    id_to_url = {}
    id_to_label = {}

    for i, url in enumerate(url_data['train']):
        train_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['train'][i]
        index += 1

    for i, url in enumerate(url_data['val']):
        val_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['val'][i]
        index += 1

    for i, url in enumerate(url_data['test_java']):
        test_java_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['test_java'][i]
        index += 1

    for i, url in enumerate(url_data['test_python']):
        test_python_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['test_python'][i]
        index += 1

    training_set = VariantOneDataset(train_ids, id_to_label, id_to_url, EMBEDDINGS_DIRECTORY)
    val_set = VariantOneDataset(val_ids, id_to_label, id_to_url, EMBEDDINGS_DIRECTORY)
    test_java_set = VariantOneDataset(test_java_ids, id_to_label, id_to_url, EMBEDDINGS_DIRECTORY)
    test_python_set = VariantOneDataset(test_python_ids, id_to_label, id_to_url, EMBEDDINGS_DIRECTORY)

    training_generator = DataLoader(training_set, **TRAIN_PARAMS)
    val_java_generator = DataLoader(val_set, **VALIDATION_PARAMS)
    test_java_generator = DataLoader(test_java_set, **TEST_PARAMS)
    test_python_generator = DataLoader(test_python_set, **TEST_PARAMS)

    model = VariantOneClassifier()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    train(model=model,
          learning_rate=LEARNING_RATE,
          number_of_epochs=NUMBER_OF_EPOCHS,
          training_generator=training_generator,
          val_generator=val_java_generator,
          test_java_generator=test_java_generator,
          test_python_generator=test_python_generator)


if __name__ == '__main__':
    do_train()
