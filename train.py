from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import pickle as pkl
import torchvision
from torchvision import models, transforms, io
from torch.utils.data import Dataset, DataLoader
import utils
import os

DATASET_PATH = 'ADE20K_2021_17_01'
index_file = 'index_ade20k.pkl'
with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
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

train_image_ids = []
validation_image_ids = []

for i in good_image_ids:
    if 'training' in index_ade20k['folder'][i]:
        train_image_ids.append(i)
    elif 'validation' in index_ade20k['folder'][i]:
        validation_image_ids.append(i)
    else:
        raise Exception('Invalid folder name.')

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

input_size = 224
transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
#                 transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

target_transform = transforms.Compose([
                transforms.Resize(input_size, interpolation=0),
                transforms.CenterCrop(input_size),
                transforms.ToTensor()
            ])

training_data = SegmentationDataset(train_image_ids[:4], './', index_ade20k, transform=transform, target_transform=target_transform)
train_dataloader = DataLoader(training_data, batch_size=2, shuffle=False)

model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=151)

# Maps {obj_ids: 0-149}
obj_id_map = {sorted(most_common_obj_ids)[idx]: idx + 1 for idx in range(150)}
obj_id_map[-1] = 0

def encode_label(label_arr):
    """
    Encode labels for evaluating loss
    label_arr (tensor): B x 1 x H x W
    """
    B, H, W = label_arr.size()[0], label_arr.size()[2], label_arr.size()[3]
    encoded_label = np.zeros((B, H, W))
    for b in range(B):
        for h in range(H):
            for w in range(W):
                class_id = label_arr[b, 0, h, w].numpy()
                new_obj_idx = obj_id_map[class_id - 1]
                encoded_label[b, h, w] = new_obj_idx
    
    return torch.tensor(encoded_label, dtype=torch.long)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()
for i in range(2):
    running_loss = 0
    for images, labels in train_dataloader:

        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        # Training pass
        optimizer.zero_grad()
        
        output = model(images)['out']
        labels = encode_label(labels)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    
    else:
        print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(train_dataloader)))













