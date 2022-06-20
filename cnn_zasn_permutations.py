import os
import csv
import torch
import numpy as np
import torchvision.transforms as transforms

from hparams import hparams
from torchsummary import summary
from cnn_training_zasn import kFoldTraining
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data import CancerDataset

def summaryModel(model, tensor_shape):
    print("Summary of a model")
    summary(model, tensor_shape)

def get_mean_std(dataset):
    '''
        Calculate mean and std for the given dataset.
    '''
    loader = DataLoader(dataset, batch_size=1)
    mean = 0.
    std = 0.
    labels = []
    for data, label in loader:
        mean += torch.mean(data)
        std += torch.std(data)
        labels.append(label.item())

    mean /= len(dataset)
    std /= len(dataset)

    return mean, std, labels

def permutate_columns(no_columns=531):

    columns = np.arange(no_columns)
    random_columns = np.random.permutation(columns)

    #np.save("Signals_permuation.npy", random_signals)

    return random_columns

def permutate_signals(no_signals=267):

    signals = np.arange(no_signals)
    random_signals = np.random.permutation(signals)

    #np.save("Signals_permuation.npy", random_signals)

    return random_signals

def save_indices(train_indices, test_indices):
    np.save("indices/Train_indices_CnC.npy", train_indices)
    np.save("indices/Test_indices_CnC.npy", test_indices)

def load_indices(train_filename, test_filename):
    train_indices = np.load(train_filename)
    test_indices = np.load(test_filename)

    return train_indices, test_indices

def load_new_split(train_filename="indices/New_train_indices_CnC.npy", test_filename="indices/New_test_indices_CnC.npy"):
    train_indices = np.load(train_filename)
    test_indices = np.load(test_filename)

    return train_indices, test_indices

def load_new_split_v2(train_filename="indices/New_disease_train_indices_CnC.npy", test_filename="indices/New_disease_test_indices_CnC.npy"):
    train_indices = np.load(train_filename)
    test_indices = np.load(test_filename)

    return train_indices, test_indices

def train_random_signal(result_filename, log_filename, signal_permutation, column_permutation, write_mode, isFold, lr_list, Dropout_list, wd_list, no_folds = 5, is_split=True):

    # std and mean normalization for transfer learning
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    cancer_type = "Cancer-nonCancer"
    data_dir = "data2/KEGG_Pathway_Image"
    annotation_file = 'annotations/Cancer_annotations_mts2.csv'
    dataset_norm = CancerDataset(annotation_file, cancer_type, data_dir)
    # normalization for our data
    mean, std, labels = get_mean_std(dataset_norm)

    normalize = transforms.Normalize(mean,std)


    dataset = CancerDataset(annotation_file,
                            cancer_type, data_dir, signal_permutation=signal_permutation, columns_permutation=column_permutation, transform=normalize)


    train_indices = None
    test_indices = None

    if not is_split:
        indices = np.arange(len(labels))
        train_indices, test_indices = train_test_split(indices, test_size=0.3, stratify=y)
        save_indices(train_indices, test_indices)
    else:
        #train_indices, test_indices = load_indices("indices/Train_indices_CnC.npy", "indices/Test_indices_CnC.npy")
        train_indices, test_indices = load_new_split()

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    y_train = np.array(labels)[train_indices]
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Get test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Count class wieghts
    no_zero_class = labels.count(0)
    no_one_class = labels.count(1)

    one_weight = no_zero_class / no_one_class
    class_weights = torch.Tensor([1, one_weight])

    # Open the file for results
    result_file = open(result_filename, write_mode)
    # Open the file for logs
    log_file = open(log_filename, write_mode)


    # Names of parameters reported in result file
    fieldnames = ["Learning_rate", "Dropout", "weight_decay", "class_weight1",
            "class_weight2", "mixup_alpha"]

    # Create columns for result of each fold
    if isFold:
        for i in range(no_folds):
            fieldnames.append("fold_" + str(i) + "_val_bal_acc")
        for i in range(no_folds):
            fieldnames.append("fold_" + str(i) + "_test_bal_acc")

    fieldnames.append("mean_val_bal_acc")
    fieldnames.append("mean_test_bal_acc")

    # Get DictWriter
    writer = csv.DictWriter(result_file, fieldnames=fieldnames)

    # Check the file mode, write headers of columns if it is a write mode
    if write_mode == "w":
        writer.writeheader()

    # Train models in k-fold training or without k-fold - depens of flag isFold
    for name in ["ResNet18"]:
        for lr in lr_list:
            for Dropout in Dropout_list:
                for wd in wd_list:
                    hparams[name]["lr"] = lr
                    hparams[name]["weight_decay"] = wd
                    hparams[name]["Dropout"] = Dropout
                    log_file.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                    log_file.write("Training on:\n")
                    log_file.write("Lr = {}\n".format(lr))
                    log_file.write("Dropout = {}\n".format(Dropout))
                    log_file.write("weight_decay = {}\n".format(wd))
                    log_file.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")

                    kFoldTraining(log_file, no_folds, writer, y_train, train_dataset, test_dataloader, test_indices, class_weights, hparams[name])

    result_file.close()
    log_file.close()

def main():

    #Define number of processes to run
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('runs'):
        os.makedirs('runs')
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('annotations'):
        os.makedirs('annotations')
    if not os.path.exists('indices'):
        os.makedirs('indices')

    #lr_list = np.logspace(-1, -3, 10)
    lr_list = [0.1]
    Dropout_list = [0.2]
    wd_list = [1e-4]

    #################### Training of signals permutation ####################
    result_file = "results/ZaSN_5_random_paths.csv"
    log_file = "logs/ZaSN_5_random_paths.txt"
    isFold = True

    signals_permutations = []
    no_signals = 5

    if not os.path.exists("{}_signals_permutations.npy".format(no_signals)):
        for i in range(no_signals):
            permutation = permutate_signals()
            signals_permutations.append(permutation)

        np.save("{}_signals_permutations.npy".format(no_signals), signals_permutations)

    else:
        signals_permutations = np.load("{}_signals_permutations.npy".format(no_signals))

    write_mode = 'w'
    #for signal_permutation in signals_permutations:

    for signal_permutation in signals_permutations:
        train_random_signal(result_file, log_file, signal_permutation, None, write_mode, isFold, lr_list, Dropout_list, wd_list)
        write_mode = 'a'


    #################### Training of columns permutation ####################
    result_file = "results/ZaSN_5_random_cols.csv"
    log_file = "logs/ZaSN_5_random_cols.txt"

    columns_permutations = []
    no_signals = 5

    if not os.path.exists("{}_columns_permutations.npy".format(no_signals)):
        for i in range(no_signals):
            permutation = permutate_columns()
            columns_permutations.append(permutation)

        np.save("{}_columns_permutations.npy".format(no_signals), columns_permutations)

    else:
        columns_permutations = np.load("{}_columns_permutations.npy".format(no_signals))

    write_mode = 'w'
    for column_permutation in columns_permutations:
        train_random_signal(result_file, log_file, None, column_permutation, write_mode, isFold, lr_list, Dropout_list, wd_list)
        write_mode = 'a'


if __name__ == "__main__":
    main()
