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
import json
import train


def get_parameter_size(model, dtype):
    """
    Return model size in terms of parameters
    Each parameter is a float32 - 4 bytes
    """
    num_params = 0
    for p in model.parameters():
        num_params += torch.count_nonzero(p.flatten())
        
    if dtype == 'float32':
        total_bytes = num_params.item() * 4
    elif dtype == 'float16':
        total_bytes = num_params.item() * 2

    kb = total_bytes / 1000
    
    return {"# Params": num_params.item(),
            "Size in KB": kb}


def quantize_weights(model, dtype=torch.float16):
    for params in list(model.parameters()):
        params.data = (params.data).type(dtype)
    
    return model


def dequantize_weights(model):    
    for params in list(model.parameters()):
        params.data = (params.data).type(torch.float32)
    
    return model
 

def validate(model, test_dataloader):
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
    # print("Testing time: {} seconds".format(time.time() - test_start))

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
    model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=151).to(device)
    model.load_state_dict(torch.load(args.weights_file, map_location=device))
    model_size = get_parameter_size(model, 'float32')
    print("Original model size: ", model_size)

    # Make copy of model
    quantized_model = copy.deepcopy(model)
    quantize_weights(quantized_model) # Convert weights to quantized dtype
    dequantize_weights(quantized_model) # Convert back to float32 for passing through network (weights themselves don't change in value)

    quantized_model_size = get_parameter_size(quantized_model, 'float16')
    
    # Retrain pruned model
    validation_metrics = validate(quantized_model, test_dataloader)
    print(validation_metrics)
    print(quantized_model_size)

    # results_dirpath = 'runs/prune/pruned_validation_results'
    # os.makedirs(results_dirpath)
    # results = {
    #     "size": pruned_model_size,
    #     "accuracies": validation_metrics
    # }
    # with open(results_dirpath + "/fcn_thresh_{}.json", 'w') as fp:
    #     json.dump(results, fp)


    