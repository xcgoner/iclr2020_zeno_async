import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os
import sys
import math
import time
import random
import datetime
import pickle

from os import listdir
import os.path
import argparse

import glob
import math

import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon.utils import download

from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data import transforms as gcv_transforms

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsplit", type=int, help="number of partitions of training set", default=20)
    parser.add_argument("--output", type=str, help="dir of output", required=True)
    args = parser.parse_args()

    output_dir = os.path.join(args.output, 'dataset_split_{}'.format(args.nsplit))
    output_train_dir = os.path.join(output_dir, 'train')
    output_val_dir = os.path.join(output_dir, 'val')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_train_dir):
        os.makedirs(output_train_dir)
    if not os.path.exists(output_val_dir):
        os.makedirs(output_val_dir)

    context = [mx.cpu()]
    batch_size = 10

    dataset_name = 'cifar-10'

    # transform_train = transforms.Compose([
    #     # Randomly crop an area, and then resize it to be 32x32
    #     transforms.RandomResizedCrop(32),
    #     # Randomly flip the image horizontally
    #     transforms.RandomFlipLeftRight(),
    #     # Randomly jitter the brightness, contrast and saturation of the image
    #     transforms.RandomColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    #     # Randomly adding noise to the image
    #     transforms.RandomLighting(0.1),
    #     # Transpose the image from height*width*num_channels to num_channels*height*width
    #     # and map values from [0, 255] to [0,1]
    #     transforms.ToTensor(),
    #     # Normalize the image with mean and standard deviation calculated across all images
    #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    # ])

    # transform_test = transforms.Compose([
    #     transforms.Resize(32),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    # ])

    transform_train = transforms.Compose([
        gcv_transforms.RandomCrop(32, pad=4),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    
    train_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='keep', num_workers=0)

    # Set train=False for validation data
    test_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, last_batch='keep', num_workers=0)

    x_train_list = []
    y_train_list = []
    for i, (data, target) in enumerate(train_data):
        x_train_list.append(data)
        y_train_list.append(target)
    x_test_list = []
    y_test_list = []
    for i, (data, target) in enumerate(test_data):
        x_test_list.append(data)
        y_test_list.append(target)

    # x_train = nd.concat(*x_train_list, dim=0)
    # y_train = nd.concat(*y_train_list, dim=0)
    # x_test = nd.concat(*x_test_list, dim=0)
    # y_test = nd.concat(*y_test_list, dim=0)

    nd.waitall()

    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)

    perm = list(range( len(x_train_list) ))
    random.shuffle(perm)
    
    x_train_list = [x_train_list[index] for index in perm]
    y_train_list = [y_train_list[index] for index in perm]

    # zeno validation
    n_val = round(len(x_train_list) / 20)
    x_val_list = []
    y_val_list = []
    for i in range(n_val):
        x_val_list.append(x_train_list.pop(0))
        y_val_list.append(y_train_list.pop(0))
    x_val = nd.concat(*x_val_list, dim=0)
    y_val = nd.concat(*y_val_list, dim=0)
    print(x_val.shape)
    print(y_val.shape)
    output_val_filename = os.path.join(output_val_dir, "val_data.pkl")
    print(output_val_filename, flush=True)
    with open(output_val_filename, "wb") as f:
        data = pickle.dumps([x_val, y_val])
        pickle.dump(data, f)

    # training data
    x_worker_list = []
    y_worker_list = []
    for i in range(args.nsplit):
        x_worker_list.append([])
        y_worker_list.append([])
    worker_index = 0
    for i in range(len(x_train_list)):
        x_worker_list[worker_index].append(x_train_list.pop(0))
        y_worker_list[worker_index].append(y_train_list.pop(0))
        worker_index = (worker_index + 1) % args.nsplit

    for i in range(args.nsplit):
        output_train_filename = os.path.join(output_train_dir, "train_data_%03d.pkl" % i)
        x_train = nd.concat(*(x_worker_list[i]), dim=0)
        y_train = nd.concat(*(y_worker_list[i]), dim=0)
        print(x_train.shape)
        print(y_train.shape)
        print(output_train_filename, flush=True)
        with open(output_train_filename, "wb") as f:
            data = pickle.dumps([x_train, y_train])
            pickle.dump(data, f)

    # testing data
    n_test = len(x_test_list)
    x_test = nd.concat(*x_test_list, dim=0)
    y_test = nd.concat(*y_test_list, dim=0)
    print(x_test.shape)
    print(y_test.shape)
    output_test_filename = os.path.join(output_val_dir, "test_data.pkl")
    print(output_test_filename, flush=True)
    with open(output_test_filename, "wb") as f:
        data = pickle.dumps([x_test, y_test])
        pickle.dump(data, f)