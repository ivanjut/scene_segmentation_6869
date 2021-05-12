from PIL import Image
import argparse
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import pickle as pkl
import torchvision
from torchvision import models, transforms, io
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import jaccard_score as IOU
import torch.nn.utils.prune as prune
import utils
import os
import sys
import time
import copy
import logging
import train


class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold


def get_parameter_size(model):
    """
    Return model size in terms of parameters
    Each parameter is a float32 - 4 bytes
    """
    num_params = 0
    for p in model.parameters():
        num_params += torch.count_nonzero(p.flatten())
        
    total_bytes = num_params.item() / 4
    kb = total_bytes / 1000
    
    return {"# Params": num_params.item(),
            "Size in KB": kb}


def train_model(model, train_dataloader, test_dataloader, obj_id_map, epochs=20, lr=0.01, momentum=0.8):
    train_start = time.time()

    ########### TODO: set up directory for storing weights
    ########### TODO: configure logging and remove print statements
    # num = len(os.listdir('runs'))+1
    # result_path = 'runs/prune_run_{}'.format(num)
    # os.makedirs(result_path)
    # logging.basicConfig(level=logging.INFO,
    #                 format='%(asctime)-8s %(message)s',
    #                 datefmt='%d %H:%M',
    #                 filename='{}/out.log'.format(result_path),
    #                 filemode='w')
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)-8s %(message)s')
    # console.setFormatter(formatter)
    # logging.getLogger('').addHandler(console)
    # logger = logging.getLogger('general_output')


    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = torch.nn.CrossEntropyLoss()
    
    for i in range(epochs):

        # logger.info('#'*30)
        # logger.info('Epoch {}'.format(i+1))
        # logger.info('#'*30)
        # epoch_start = time.time()

        # training pass
        running_loss = 0
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)['out']
            labels = train.encode_label(labels, obj_id_map).to(device)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print("Batch finished...")
            # logger.info('Batch finished...')
        # logger.info('Training loss: {}'.format(running_loss/len(train_dataloader)))
        # logger.info("Training time: {} seconds".format(time.time() - epoch_start))
        # torch.save(model.state_dict(), result_path+'/epochs_{}_weights.pkl'.format(i+1))

        # testing pass
        test_start = time.time()
        running_accuracy = 0
        running_iou = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                output = model(images)['out']
                labels = train.encode_label(labels, obj_id_map).to(device)
                probs = torch.nn.functional.softmax(output, dim=1)
                preds = torch.argmax(probs, dim=1, keepdim=True).squeeze()
                num_correct = torch.sum((preds == labels).to(int)).item()
                iou = IOU(labels.detach().cpu().numpy().reshape(-1), preds.detach().cpu().numpy().reshape(-1), average='weighted')
                # logger.info('Testing accuracy: {}'.format(num_correct/(224*224*len(images))))
                # logger.info('Testing IOU score: {}'.format(iou))
                running_accuracy += num_correct/(224*224*len(images))
                running_iou += iou
    #     logger.info("Testing time: {} seconds".format(time.time() - test_start))
    #     logger.info('-----> Overall testing pixel accuracy: {}'.format(running_accuracy / len(test_dataloader)))
    #     logger.info('-----> Overall testing IOU accuracy: {}'.format(running_iou / len(test_dataloader)))
    #     logger.info("Epoch completed in {} seconds.".format(time.time() - epoch_start))
    
    # logger.info('#'*30)
    # logger.info("DONE TRAINING in {} seconds.".format(time.time() - train_start))
    # logger.info('#'*30)

    print("DONE TRAINING in {} seconds.".format(time.time() - train_start))
    return model


def validate(model, test_dataloader):
    ######## TODO: once logging is set up, remove print statements
    # testing pass
    test_start = time.time()
    running_accuracy = 0
    running_iou = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            output = model(images)['out']
            labels = train.encode_label(labels, obj_id_map).to(device)
            probs = torch.nn.functional.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1, keepdim=True).squeeze()
            num_correct = torch.sum((preds == labels).to(int)).item()
            iou = IOU(labels.detach().cpu().numpy().reshape(-1), preds.detach().cpu().numpy().reshape(-1), average='weighted')
            # logger.info('Testing accuracy: {}'.format(num_correct/(224*224*len(images))))
            # logger.info('Testing IOU score: {}'.format(iou))
            running_accuracy += num_correct/(224*224*len(images))
            running_iou += iou
    # logger.info("Testing time: {} seconds".format(time.time() - test_start))
    print("Testing time: {} seconds".format(time.time() - test_start))

    return {"Testing pixel accuracy": running_accuracy / len(test_dataloader),
            "Testing IOU accuracy": running_iou / len(test_dataloader)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='gpu', type=str, help='Whether you are using GPU or CPU')
    parser.add_argument('-weights_file', default='', type=str, help='Path to weights pickle file')
    # parser.add_argument('-thresh', default=0.1, type=float, help='Threshold of weights to prune')
    parser.add_argument('-batch_size', default=32, type=int, help='Number of samples in a batch')
    args = parser.parse_args()

    device = torch.device('cuda:0' if args.device == 'gpu' else 'cpu')
    load_data_start = time.time()
    train_dataloader, test_dataloader, obj_id_map = train.get_data(args.batch_size)
    print("Loaded data. ({} sec.)".format(time.time() - load_data_start))

    # Load trained weights from pkl file
    model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=151).to(device)
    model.load_state_dict(torch.load('../epochs_20_weights.pkl', map_location=device))
    model_size = get_parameter_size(model)
    print("Original model size: ", model_size)

    thresholds = [0.001, 0.0025, 0.01, 0.025, 0.1]
    for thresh in thresholds:

        # Make copy of model
        model_to_prune = copy.deepcopy(model)
        # matching_flag = True
        # for p1, p2 in zip(model.parameters(), model_to_prune.parameters()):
        #     if p1.data.ne(p2.data).sum() > 0:
        #         matching_flag = False
        # assert matching_flag is True

        # Prune
        params_to_prune = [(module, "weight") for _, module in model_to_prune.named_modules() if isinstance(module, torch.nn.Conv2d)]

        prune.global_unstructured(params_to_prune, pruning_method=ThresholdPruning, threshold=thresh)

        for _, module in model_to_prune.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.remove(module, 'weight')

        pruned_model_size = get_parameter_size(model_to_prune)
        
        # Retrain pruned model
        model_to_prune = train_model(model_to_prune, train_dataloader, test_dataloader, obj_id_map)
        validation_metrics = validate(model_to_prune, test_dataloader)


        ########## TODO: save these to JSON? - can remove print statements for logging
        print("*" * 10, "Threshold {}".format(thresh), "*" * 10)
        print("SIZE: ", pruned_model_size)
        print("ACCURACIES: ", validation_metrics)
        #######################################

    