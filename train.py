from PIL import Image
import argparse
import json
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
import utils
import os
import sys
import time
import logging

class SegmentationDataset(Dataset):
    def __init__(self, image_ids, root_dir, index_mat, transform=None, target_transform=None):
        """
        Args:
            image_ids (list): list of image IDs from ADE20K
            root_dir (string): Directory with all the images.
            index_mat (array): object array from index_ade20k.pkl
            transform (callable, optional): Optional transform to be applied
                on a sample.
            target_transform (callable, optional): Optional transform to be applied
                on a sample segmentation label.
        """
        self.image_ids = image_ids
        self.root_dir = root_dir
        self.index_ade20k = index_mat
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_name = os.path.join(self.root_dir, self.index_ade20k['folder'][image_id], 
                                self.index_ade20k['filename'][image_id])
        img_info = utils.loadAde20K(img_name)
        image = io.read_image(img_info['img_name']).float()
        class_mask = Image.fromarray(img_info['class_mask'], mode='I')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(class_mask)
        sample = (image, label)
        return sample


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


def get_data(batch_size):
    """
    Generate training/testing data loaders
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

    train_image_ids = []
    test_image_ids = []
    for i in good_image_ids:
        if 'training' in index_ade20k['folder'][i]:
            train_image_ids.append(i)
        elif 'validation' in index_ade20k['folder'][i]:
            test_image_ids.append(i)
        else:
            raise Exception('Invalid folder name.')

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

    training_data = SegmentationDataset(train_image_ids[:4], './', index_ade20k, transform=transform, target_transform=target_transform)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
    testing_data = SegmentationDataset(test_image_ids[:4], './', index_ade20k, transform=transform, target_transform=target_transform)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, obj_id_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='gpu', type=str, help='Whether you are using GPU or CPU')
    parser.add_argument('-epochs', default=10, type=int, help='Number of learning epochs')
    parser.add_argument('-lr', default=0.01, type=float, help='The learning rate to use')
    parser.add_argument('-momentum', default=0.0, type=float, help='The momentum factor to use')
    parser.add_argument('-batch_size', default=32, type=int, help='Number of samples in a batch')
    parser.add_argument('-model', type=str, help='Model class to use')
    args = parser.parse_args()

    device = torch.device('cuda:0' if args.device == 'gpu' else 'cpu')
    load_data_start = time.time()
    train_dataloader, test_dataloader, obj_id_map = get_data(args.batch_size)
    print("Loaded data. ({} sec.)".format(time.time() - load_data_start))

    num = len(os.listdir('runs'))+1
    result_path = 'runs/run_{}'.format(num)
    os.makedirs(result_path)
    with open('{}/args.json'.format(result_path), 'w') as argfile:
        json.dump(vars(args), argfile)
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
    
    if args.model == 'fcn_resnet_50':
        model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=151).to(device)
    elif args.model == 'fcn_resnet_101':
        model = models.segmentation.fcn_resnet101(pretrained=False, num_classes=151).to(device)
    elif args.model == 'deeplab_resnet_50':
        model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=151).to(device)
    elif args.model == 'deeplab_resnet_101':
        model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=151).to(device)
    elif args.model == 'deeplab_mobilenet_v3_large':
        model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=False, num_classes=151).to(device)
    elif args.model == 'lraspp':
        model = models.segmentation.lraspp_mobilenet_v3_large(pretrained=False, num_classes=151).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()
    
    batches = 0
    for i in range(args.epochs):

        logger.info('#'*30)
        logger.info('Epoch {}'.format(i+1))
        logger.info('#'*30)
        epoch_start = time.time()

        # training pass
        running_loss = 0
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)['out']
            labels = encode_label(labels, obj_id_map).to(device)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            logger.info('Batch loss: {}'.format(loss.item()))
            writer.add_scalar('Batch loss', loss.item(), batches)
            batches += 1
        logger.info('Training loss: {}'.format(running_loss/len(train_dataloader)))
        logger.info("Training time: {} seconds".format(time.time() - epoch_start))
        writer.add_scalar('Epoch loss', running_loss/len(train_dataloader), i)
        torch.save(model.state_dict(), result_path+'/epochs_{}_weights.pkl'.format(i+1))

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
    logger.info("DONE TRAINING in {} seconds.".format(time.time() - load_data_start))
    logger.info('#'*30)
