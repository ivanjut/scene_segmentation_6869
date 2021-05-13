from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.metrics import jaccard_score as IOU
from torchvision import models, transforms, io
from torch.utils.data import Dataset, DataLoader
import utils
import os
import time
import copy
import argparse
import train


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_validation_data(batch_size):
    """
    Generate testing data loaders
    """
    with open('ADE20K_2021_17_01/index_ade20k.pkl', 'rb') as f:
        index_ade20k = pkl.load(f)
    objects_mat = index_ade20k['objectPresence']

    # Find 150 most common object IDs and non-common object IDs
    total_object_counts = np.sum(objects_mat, axis=1)
    object_count_ids = np.argsort(total_object_counts)[::-1]
    most_common_obj_ids = object_count_ids[:150]
    irrelevant_obj_ids = object_count_ids[150:]

    # Find image IDs where no irrelevant objects appear
    irrelevant_obj_counts = np.sum(objects_mat[irrelevant_obj_ids], axis=0)
    good_image_ids = np.argwhere(irrelevant_obj_counts == 0).flatten()

    # Only common objects included
    common_objects_mat = objects_mat[np.ix_(most_common_obj_ids, good_image_ids)]

    obj_id_map = {sorted(most_common_obj_ids)[idx]: idx + 1 for idx in range(150)}
    obj_id_map[-1] = 0

    test_image_ids = []
    for i in good_image_ids:
        if 'validation' in index_ade20k['folder'][i]:
            test_image_ids.append(i)

    input_size = 224
    transform = transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    target_transform = transforms.Compose([
                    transforms.Resize(input_size, interpolation=0),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor()
                ])

    testing_data = train.SegmentationDataset(test_image_ids, './', index_ade20k, transform=transform, target_transform=target_transform)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)

    return test_dataloader, obj_id_map


def encode_label(label_arr, obj_id_map):
    """
    Encode labels for evaluating loss
    label_arr (tensor): B x 1 x H x W
    obj_id_map: dictionary mapping label class IDs to new (0-150) range IDs
    """
    convert_label_ids = lambda i: obj_id_map[i-1]
    vect_convert_label_ids = np.vectorize(convert_label_ids)
    encoded_label = vect_convert_label_ids(label_arr.squeeze().cpu().numpy())
    
    return torch.tensor(encoded_label, dtype=torch.long)


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


def validate(model, test_dataloader, obj_id_map):
    # testing pass
    test_start = time.time()
    running_accuracy = 0
    running_iou = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            output = model(images)['out']
            labels = encode_label(labels, obj_id_map).to(device)
            probs = torch.nn.functional.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1, keepdim=True).squeeze()
            num_correct = torch.sum((preds == labels).to(int)).item()
            iou = IOU(labels.detach().cpu().numpy().reshape(-1), preds.detach().cpu().numpy().reshape(-1), average='weighted')
            print('Testing accuracy: {}'.format(num_correct/(224*224*len(images))))
            print('Testing IOU score: {}'.format(iou))
            running_accuracy += num_correct/(224*224*len(images))
            running_iou += iou

    print("Testing time: {} seconds".format(time.time() - test_start))

    return {"Testing pixel accuracy": running_accuracy / len(test_dataloader),
            "Testing IOU accuracy": running_iou / len(test_dataloader)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-device', default='gpu', type=str, help='Whether you are using GPU or CPU')
    parser.add_argument('-batch_size', default=32, type=int, help='Number of samples in a batch')
    parser.add_argument('-model', type=str, help='Model class to use')
    args = parser.parse_args()

    test_dataloader, obj_id_map = get_validation_data(args.batch_size)

    model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=151).to(device=device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    validation_metrics = validate(model, test_dataloader, obj_id_map)
    print(validation_metrics)

    model_size = get_parameter_size(model)
    print(model_size)


