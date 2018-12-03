import gc
import glob
import os
from pathlib import Path
from utils import TGSSaltDataset
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import models
import argparse

parser=argparse.ArgumentParser


device = 'cuda:0'
data_src = '../input/'

quick_try = False
grayscale = False

orig_image_size = (101, 101)
image_size = (128, 128)

print('Initialize.')

train_df = pd.read_csv('{}train.csv'.format(data_src),
                       usecols=[0], index_col='id')
depths_df = pd.read_csv('{}depths.csv'.format(data_src),
                        index_col='id')

train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

X_train = []
y_train = []

print('Loading training set.')
for i in tqdm.tqdm(train_df.index):
    img_src = '{}train/images/{}.png'.format(data_src, i)
    mask_src = '{}train/masks/{}.png'.format(data_src, i)
    if grayscale:
        img_temp = cv2.imread(img_src, 0)
    else:
        img_temp = cv2.imread(img_src)
    mask_temp = cv2.imread(mask_src, 0)
    if orig_image_size != image_size:
        img_temp = cv2.resize(img_temp, image_size)
        mask_temp = cv2.resize(mask_temp, image_size)
    X_train.append(img_temp)
    y_train.append(mask_temp)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
if grayscale:
    X_train = np.expand_dims(X_train, -1)
y_train = np.expand_dims(y_train, -1)

print('Compute mask coverage for each observation.')

def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i

# Percent of area covered by mask.
train_df['coverage'] = np.mean(y_train / 255., axis=(1, 2))
train_df['coverage_class'] = train_df.coverage.map(
    cov_to_class)


# del X_train, y_train
# gc.collect()

train_path = data_src + 'train'
test_path = data_src

train_ids = train_df.index.values
test_ids = test_df.index.values

from sklearn.model_selection import train_test_split

tr_ids, valid_ids, tr_coverage, valid_coverage = train_test_split(
    train_ids,
    train_df.coverage.values,
    test_size=0.2, stratify=train_df.coverage_class, random_state= 1234)

dataset_train = TGSSaltDataset(train_path, tr_ids, divide=True)
dataset_train.set_padding()
y_min_pad, y_max_pad, x_min_pad, x_max_pad = dataset_train.return_padding_borders()

# Validation dataset:
dataset_val = TGSSaltDataset(train_path, valid_ids, divide=True)
dataset_val.set_padding()

# Test dataset:
dataset_test = TGSSaltDataset(test_path, test_ids, is_test=True, divide=True)
dataset_test.set_padding()

# Data loaders:
# Use multiple workers to optimize data loading speed.
# Pin memory for quicker GPU processing.
train_loader = data.DataLoader(
    dataset_train,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True)

# Do not shuffle for validation and test.
valid_loader = data.DataLoader(
    dataset_val,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True)

test_loader = data.DataLoader(
    dataset_test,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True)

model = get_model({'num_filters': 32})
# Set Binary Crossentropy as loss function.
loss_fn = torch.nn.BCELoss()

# Set optimizer.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train for n epochs
n = 2
for e in range(n):

    # Training:
    train_loss = []
    for image, mask in tqdm.tqdm(train_loader):
        # Put image on chosen device
        image = image.type(torch.float).to(device)
        # Predict with model:
        y_pred = model(image)
        # Compute loss between true and predicted values
        loss = loss_fn(y_pred, mask.to(device))

        # Set model gradients to zero.
        optimizer.zero_grad()
        # Backpropagate the loss.
        loss.backward()

        # Perform single optimization step - parameter update
        optimizer.step()

        # Append training loss
        train_loss.append(loss.item())

    # Validation:
    val_loss = []
    val_iou = []
    for image, mask in valid_loader:
        image = image.to(device)
        y_pred = model(image)

        loss = loss_fn(y_pred, mask.to(device))
        val_loss.append(loss.item())

    print("Epoch: %d, Train: %.3f, Val: %.3f" %
          (e, np.mean(train_loss), np.mean(val_loss)))


val_predictions = []
val_masks = []

for image, mask in tqdm.tqdm(valid_loader):
    image = image.type(torch.float).to(device)
    # Put prediction on CPU, detach it and transform to a numpy array.
    y_pred = model(image).cpu().detach().numpy()
    val_predictions.append(y_pred)
    val_masks.append(mask)


# Stack all masks and predictions along first axis.
# Output of valid_loader is of shape (NxBxCxHxW), where N is number of batches and B is batch size.
val_predictions_stacked = np.vstack(val_predictions)[:, 0, :, :]
val_masks_stacked = np.vstack(val_masks)[:, 0, :, :]


# Cut off padded parts of images.
val_predictions_stacked = val_predictions_stacked[
    :, y_min_pad:-y_max_pad, x_min_pad:-x_max_pad]

val_masks_stacked = val_masks_stacked[
    :, y_min_pad:-y_max_pad, x_min_pad:-x_max_pad]

print(val_masks_stacked.shape, val_predictions_stacked.shape)

random_index = np.random.randint(0, val_masks_stacked.shape[0])
print('Validation Index: {}'.format(random_index))

fig, ax = plt.subplots(2, 1)
ax[0].imshow(val_masks_stacked[random_index], cmap='seismic')
ax[1].imshow(val_predictions_stacked[random_index] > 0.5, cmap='seismic')

test_predictions = []

for image in tqdm.tqdm(test_loader):
    image = image[0].type(torch.float).to(device)
    y_pred = model(image).cpu().detach().numpy()
    test_predictions.append(y_pred)

test_predictions_stacked = np.vstack(test_predictions)[:, 0, :, :]
test_predictions_stacked = test_predictions_stacked[:, y_min_pad:-y_max_pad, x_min_pad:-x_max_pad]

print(test_predictions_stacked.shape)