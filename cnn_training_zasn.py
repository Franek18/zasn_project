import time
import copy
import torch
import numpy as np
from torch import nn
from resnet_1d import resnet18
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_cnt = 0

def get_ResNet(dropout, no_layers=18, pretrained=False):
    '''
    '''
    global device
    resnet = None
    # Choose no of layers in ResNet
    resnet = resnet18(pretrained=pretrained)
    num_ftrs = resnet.fc.in_features
    # Here the size of each output sample is set to 2.
    resnet.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_ftrs, 2)
    )

    resnet = resnet.to(device)
    return resnet

def calc_confusion_matrix(conf_mtx, preds, labels):

    for pred, label in zip(preds, labels):
        if pred == 1 and label == 1:
            conf_mtx["TP"] += 1

        elif pred == 1 and label == 0:
            conf_mtx["FP"] += 1

        elif pred == 0 and label == 0:
            conf_mtx["TN"] += 1

        else:
            conf_mtx["FN"] += 1

def mixup_data(x, y, alpha):
    '''Returns mixed inputs and targets'''

    lambda_p = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lambda_p * x + (1 - lambda_p) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lambda_p

def mixup_criterion(criterion, pred, y_a, y_b, lambda_p):
    return (lambda_p * criterion(pred, y_a) + (1 - lambda_p) * criterion(pred, y_b))

def train_model(run_writer, log_file, model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, mixup, alpha):
    since = time.time()

    best_model = copy.deepcopy(model.state_dict())
    # validation balanced accuracy
    best_bal_acc = 0.0

    best_epoch = 0

    global device

    all_loss = []
    all_bal_acc = []
    for epoch in range(num_epochs):
        log_file.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
        log_file.write('--------------------\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            all_outputs = []
            all_preds = []
            all_labels = []
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # confusion matrix
            conf_mtx = {
                "TP" : 0,
                "FP" : 0,
                "TN" : 0,
                "FN" : 0
            }

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                if phase == 'train' and mixup:
                    inputs, labels_a, labels_b, lambda_p = mixup_data(inputs, labels, alpha)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    max_outputs = outputs[:, 1]
                    if phase == 'train' and mixup:
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lambda_p)
                    else:
                        loss = criterion(outputs, labels)
                        calc_confusion_matrix(conf_mtx, preds, labels)

                    # backward + optimize only if in training phase
                    all_outputs.extend(max_outputs.detach().cpu().numpy())
                    all_preds.extend(preds.detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            all_outputs = np.array(all_outputs, dtype=float)
            all_preds = np.array(all_preds, dtype=int)
            all_labels = np.array(all_labels, dtype=int)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train' and mixup == True:
                log_file.write('{} Loss: {:.4f}\n'.format(phase, epoch_loss))
                run_writer.add_scalar("Loss/{}_mixup_alpha_{}".format(phase, alpha), epoch_loss, epoch)

            if phase == 'val' or (phase == 'train' and mixup == False):
                try:
                    curr_recall = conf_mtx["TP"] / (conf_mtx["TP"] + conf_mtx["FN"])
                except ZeroDivisionError:
                    curr_recall = 0

                try:
                    curr_specificity = conf_mtx["TN"] / (conf_mtx["TN"] + conf_mtx["FP"])
                except ZeroDivisionError:
                    curr_specificity = 0

                bal_acc = (curr_recall + curr_specificity) / 2

                auc = roc_auc_score(all_labels, all_outputs, average='weighted')

                log_file.write('Confusion matrix: {}\n'.format(conf_mtx))
                log_file.write('{} Loss: {:.4f} Recall: {:.4f} Specificity: {:.4f} Bal Acc: {:.4f} Auc: {:.4f}\n'.format(
                    phase, epoch_loss, curr_recall, curr_specificity, bal_acc, auc))

                all_loss.append(epoch_loss)
                all_bal_acc.append(bal_acc)

                if phase == 'val' and (bal_acc > best_bal_acc):
                    # deep copy the model
                    best_epoch = epoch
                    best_bal_acc = bal_acc
                    best_model = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    #print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(best_model)

    return model, best_bal_acc, best_epoch, all_loss, all_bal_acc


def kFoldTraining(log_file, no_folds, writer, y, train_dataset, test_dataloader, test_indices, class_weights, hparams):
    # Stratified K Fold (by default 5-fold)
    skf = StratifiedKFold()
    global train_cnt
    row = { "Learning_rate": hparams["lr"], "Dropout": hparams["Dropout"],
            "weight_decay": hparams["weight_decay"], "class_weight1": class_weights[0].item(),
            "class_weight2": class_weights[1].item(), "mixup_alpha": hparams["mixup_alpha"]}

    val_bal_accs = []
    test_bal_accs = []
    all_bal_acc = []
    all_loss = []
    #fig, axes = plt.subplots(2, 3)
    #fig, axes = plt.subplots()
    num_epochs=hparams["num_of_epochs"]
    lr = hparams["lr"]
    Dropout = hparams["Dropout"]
    wd = hparams["weight_decay"]
    train_cnt += 1
    run_dir=f"runs/ZaSN3_permutations_{train_cnt}_lr_{lr}_Dropout_{Dropout}_wd_{wd}"
    run_writer = SummaryWriter(log_dir=run_dir)
    for fold, (train_ids, val_ids) in enumerate(skf.split(np.zeros(len(y)), y)):
        log_file.write(f'FOLD {fold}\n')
        log_file.write('--------------------------------\n')
        # Get the model
        model = get_ResNet(hparams["Dropout"], no_layers=hparams["no_layers"], pretrained=hparams["pretrained"])
        # Loss function and optimizer
        loss_fn = nn.CrossEntropyLoss(class_weights.to(device))
        optimizer = torch.optim.SGD(model.parameters(), lr=hparams["lr"],
                                    weight_decay = hparams["weight_decay"])
        # and SteLR
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=hparams["step_size"],
                                                gamma=hparams["gamma"])

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_size = len(train_ids)
        val_size = len(val_ids)
        #val_size = len(test_indices)

        dataset_sizes = {'train': train_size, 'val': val_size}

        # Get dataloaders
        trainloader = DataLoader(train_dataset, batch_size=hparams["train_batch_size"], sampler=train_subsampler)
        validloader = DataLoader(train_dataset, batch_size=hparams["val_batch_size"], sampler=val_subsampler)

        dataloaders = {'train': trainloader, 'val': validloader}

        # train
        best_model, val_bal_acc, model_epoch, fold_loss, fold_bal_acc  = train_model(run_writer,
                                                            log_file,
                                                            model,
                                                            dataloaders,
                                                            dataset_sizes,
                                                            loss_fn,
                                                            optimizer,
                                                            exp_lr_scheduler,
                                                            num_epochs, mixup=hparams["mixup"], alpha=hparams["mixup_alpha"])


        all_loss.append(fold_loss)
        all_bal_acc.append(fold_bal_acc)
        # test
        test_loss, test_bal_acc = test(log_file, test_dataloader, len(test_indices), best_model, nn.CrossEntropyLoss())

        # append val and test bal_acc from current fold
        val_bal_accs.append(val_bal_acc)
        test_bal_accs.append(test_bal_acc)



    all_loss = np.array(all_loss)
    all_bal_acc = np.array(all_bal_acc)

    for i in range(no_folds):
        new_val_key = "fold_" + str(i) + "_val_bal_acc"
        row[new_val_key] = val_bal_accs[i]

    for i in range(no_folds):
        new_test_key = "fold_" + str(i) + "_test_bal_acc"
        row[new_test_key] = test_bal_accs[i]

    row['mean_val_bal_acc'] = np.mean(val_bal_accs)
    row['mean_test_bal_acc'] = np.mean(test_bal_accs)
    # save the result row to the csv file
    writer.writerow(row)

    iter = {'train': 0, 'val': 1}
    for j in range(0, 2*num_epochs, 2):
        for phase in ['train', 'val']:
            z = iter[phase]
            cv_loss = 0
            cv_bal_acc = 0
            for i in range(no_folds):
                cv_loss += all_loss[i, j + z]
                cv_bal_acc += all_bal_acc[i, j + z]

            cv_loss /= no_folds
            cv_bal_acc /= no_folds
            epoch = j // 2 + 1
            run_writer.add_scalar("Loss/{}".format(phase), cv_loss, epoch)
            run_writer.add_scalar("Bal_acc/{}".format(phase), cv_bal_acc, epoch)


    run_writer.flush()
    run_writer.close()


def test(log_file, dataloader, size, model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    # confusion matrix
    conf_mtx = {
        "TP" : 0,
        "FP" : 0,
        "TN" : 0,
        "FN" : 0
    }

    all_outputs = []
    all_preds = []
    all_labels = []
    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            max_outputs = outputs[:, 1]
            loss = criterion(outputs, labels)
            calc_confusion_matrix(conf_mtx, preds, labels)

        all_outputs.extend(max_outputs.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    all_outputs = np.array(all_outputs, dtype=float)
    all_preds = np.array(all_preds, dtype=int)
    all_labels = np.array(all_labels, dtype=int)

    try:
        curr_recall = conf_mtx["TP"] / (conf_mtx["TP"] + conf_mtx["FN"])
    except ZeroDivisionError:
        curr_recall = 0

    try:
        curr_specificity = conf_mtx["TN"] / (conf_mtx["TN"] + conf_mtx["FP"])
    except ZeroDivisionError:
        curr_specificity = 0

    test_loss = running_loss / size
    test_bal_acc = (curr_recall + curr_specificity) / 2
    test_auc = roc_auc_score(all_labels, all_outputs, average='weighted')

    log_file.write('Test Confusion matrix: {}\n'.format(conf_mtx))
    log_file.write('Test Loss: {:.4f} Recall: {:.4f} Specificity: {:.4f} Bal Acc: {:.4f} Auc: {:.4f}\n'.format(
            test_loss, curr_recall, curr_specificity, test_bal_acc, test_auc))

    return test_loss, test_bal_acc
