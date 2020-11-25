import copy
import os
import random

from PIL import Image

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

import numpy as np
import pandas as pd


class MyCustomDataset(Dataset):
    def __init__(self, dict_path):
        self.transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.class_dict = pd.read_csv(dict_path)

    def __getitem__(self, index):
        row = self.class_dict.iloc[index]
        file_path = row[1]
        data = Image.open(file_path).convert("RGB").resize(IMAGE_NORMAL_SIZE)
        data = self.transformations(data)  # (3)
        label = row[2]
        return data, label

    def __len__(self):
        return self.class_dict.index.shape[0]


class StyleMatrix(nn.Module):
    def __init__(self):
        super(StyleMatrix, self).__init__()

    def forward(self, input_):
         return __class__.gram_matrix(input_)

    @staticmethod
    def gram_matrix(inp):
        a, b, c, d = inp.size()
        features = inp.view(a, b, c * d)
        G = torch.empty((a, b, b))
        for i in range(a):
            G[i] = torch.mm(features[i], features[i].t())
        return G.div(a * b * c * d)


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            index = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if (100 * index // len(dataloaders[phase])) // 10 != (100 * (index - 1) // len(dataloaders[phase])) // 10:
                    print('=', end='')
                index += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('\t{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract=True, add_gram=False, n_layers=None,
                     use_pretrained=True, classifier=None):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        if n_layers is not None:
            if add_gram == False:
                model_ft = nn.Sequential(*list(model_ft.children())[:n_layers])
            else:
                model_ft = nn.Sequential(*(list(model_ft.children())[:n_layers]) + [StyleMatrix()])
        if classifier is not None:
            model_ft = nn.Sequential(*(list(model_ft.children())) + [classifier])

    elif model_name == "vgg19_bn":
        """ VGG-19 with batch normalization
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        if n_layers is not None:
            if add_gram == False:
                model_ft = nn.Sequential(*list(model_ft.features.children())[:n_layers])
            else:
                model_ft = nn.Sequential(*(list(model_ft.features.children())[:n_layers]) + [StyleMatrix()])
        if classifier is not None:
            model_ft.classifier = classifier

    elif model_name == "vgg19":
        """ VGG-19
        """
        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        if n_layers is not None:
            if add_gram == False:
                model_ft = nn.Sequential(*list(model_ft.features.children())[:n_layers])
            else:
                model_ft = nn.Sequential(*(list(model_ft.features.children())[:n_layers]) + [StyleMatrix()])
        if classifier is not None:
            model_ft.classifier = classifier

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft.to(DEVICE), input_size
