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
from torch.utils.tensorboard import SummaryWriter
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
        num_params += torch.count_nonzero(p.flatten().detach())
        
    total_bytes = num_params.item() * 4
    kb = total_bytes / 1000
    
    return {"# Params": num_params.item(),
            "Size in KB": kb}


def train_model(model, teacher_models, train_dataloader, test_dataloader, obj_id_map, device, epochs=20, lr=0.01, momentum=0.8):
    train_start = time.time()

    # TODO: configure save directory - I messed this up last time
    num = len(os.listdir('runs/kd/'))+1
    result_path = 'runs/kd/kd_run_{}'.format(num)
    os.makedirs(result_path)
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-8s %(message)s',
                    datefmt='%d %H:%M',
                    filename='{}/out.log'.format(result_path),
                    filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger('general_output')
    writer = SummaryWriter(log_dir=result_path)

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion_soft = torch.nn.CrossEntropyLoss()
    criterion_hard = torch.nn.CrossEntropyLoss()
    
    batches = 0
    for i in range(epochs):

        logger.info('#'*30)
        logger.info('Epoch {}'.format(i+1))
        logger.info('#'*30)
        epoch_start = time.time()

        # training pass
        running_loss = 0
        for images, labels in train_dataloader:
            images = images.to(device)
            optimizer.zero_grad()

            shape, full_tensor = None, None
            for tm in teacher_models:
                soft_label = tm(images)['out']
                if shape is None:
                    shape = soft_label.size()
                    full_tensor = torch.zeros(shape).to(device)
                full_tensor = full_tensor.add(soft_label)
            soft_label_outputs = torch.div(full_tensor, len(teacher_models))
            label_probs = torch.nn.functional.softmax(soft_label_outputs, dim=1)
            soft_labels = torch.argmax(label_probs, dim=1, keepdim=True).squeeze()

            output = model(images)['out']
            loss_soft = criterion_soft(output, soft_labels.to(device))
            loss_hard = criterion_hard(output, train.encode_label(labels, obj_id_map).to(device))
            loss = sum([loss_soft * 0.5, loss_hard * 0.5])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            logger.info('Batch finished: {}'.format(loss.item()))
            writer.add_scalar('Batch loss', loss.item(), batches)
            batches += 1
        logger.info('Training loss: {}'.format(running_loss/len(train_dataloader)))
        logger.info("Training time: {} seconds".format(time.time() - epoch_start))
        writer.add_scalar('Epoch loss', running_loss/len(train_dataloader), i)
        if i+1 == 20:
            torch.save(model, result_path+'/epochs_{}_model.pkl'.format(i+1))

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
                logger.info('Testing accuracy: {}'.format(num_correct/(224*224*len(images))))
                logger.info('Testing IOU score: {}'.format(iou))
                running_accuracy += num_correct/(224*224*len(images))
                running_iou += iou
        logger.info("Testing time: {} seconds".format(time.time() - test_start))
        logger.info('-----> Overall testing pixel accuracy: {}'.format(running_accuracy / len(test_dataloader)))
        logger.info('-----> Overall testing IOU accuracy: {}'.format(running_iou / len(test_dataloader)))
        writer.add_scalar('Pixel acc', running_accuracy/len(test_dataloader), i)
        writer.add_scalar('IOU', running_iou/len(test_dataloader), i)
        epoch_duration = time.time() - epoch_start
        logger.info("Epoch completed in {} seconds.".format(epoch_duration))
        writer.add_scalar('Epoch duration', epoch_duration, i)
    
    logger.info('#'*30)
    logger.info("DONE TRAINING in {} seconds.".format(time.time() - train_start))
    logger.info('#'*30)

    return model, result_path


def validate(model, test_dataloader, device):
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
            logger.info('Testing accuracy: {}'.format(num_correct/(224*224*len(images))))
            logger.info('Testing IOU score: {}'.format(iou))
            running_accuracy += num_correct/(224*224*len(images))
            running_iou += iou
    logger.info("Testing time: {} seconds".format(time.time() - test_start))

    return {"Testing pixel accuracy": running_accuracy / len(test_dataloader),
            "Testing IOU accuracy": running_iou / len(test_dataloader)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='gpu', type=str, help='Whether you are using GPU or CPU')
    # parser.add_argument('-model', default='fcn_resnet_50', type=str, help='Model to train with KD')
    # parser.add_argument('-thresh', default=0.1, type=float, help='Threshold of weights to prune')
    parser.add_argument('-batch_size', default=32, type=int, help='Number of samples in a batch')
    args = parser.parse_args()

    device = torch.device('cuda:0' if args.device == 'gpu' else 'cpu')
    load_data_start = time.time()
    train_dataloader, test_dataloader, obj_id_map = train.get_data(args.batch_size)
    print("Loaded data. ({} sec.)".format(time.time() - load_data_start))

    # Load teacher models
    teacher_model_1 = models.segmentation.fcn_resnet50(pretrained=False, num_classes=151).to(device)
    teacher_model_1.load_state_dict(torch.load('./models/fcn_resnet_50/epochs_20_weights.pkl', map_location=device))
    teacher_model_2 = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=151).to(device)
    teacher_model_2.load_state_dict(torch.load('./models/deeplab_resnet_50/epochs_20_weights.pkl', map_location=device))
    teacher_models = [teacher_model_1, teacher_model_2]

    # Load pruned model architecture for retraining
    model = copy.deepcopy(teacher_model_1) # fcn resnet 50
    params_to_prune = [(module, "weight") for _, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]
    prune.global_unstructured(params_to_prune, pruning_method=ThresholdPruning, threshold=0.025)
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')

    model, result_path = train_model(model, teacher_models, train_dataloader, test_dataloader, obj_id_map, device=device)

    model_size = get_parameter_size(model)
        
    validation_metrics = validate(model, test_dataloader, device)

    print(model_size)
    print(validation_metrics)
    results = {
        "size": model_size,
        "accuracies": validation_metrics
    }
    with open(result_path + "/fcn_kd.json", 'w') as fp:
        json.dump(results, fp)


    
