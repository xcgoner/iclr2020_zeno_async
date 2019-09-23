import argparse, time, logging, os, math, random
os.environ["MXNET_USE_OPERATOR_TUNING"] = "0"


import numpy as np
from scipy import stats
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRScheduler

from os import listdir
import os.path
import argparse

import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, help="dir of the data", required=True)
parser.add_argument("--batchsize", type=int, help="batchsize", default=128)
parser.add_argument("--epochs", type=int, help="number of epochs", default=100)
parser.add_argument("--interval", type=int, help="log interval (epochs)", default=10)
parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
parser.add_argument("--lr-decay", type=float, help="lr decay rate", default=0.5)
parser.add_argument("--lr-decay-epoch", type=str, help="lr decay epoch", default='2000')
parser.add_argument("--momentum", type=float, help="momentum", default=0)
parser.add_argument("--log", type=str, help="dir of the log file", default='train_cifar100.log')
parser.add_argument("--classes", type=int, help="number of classes", default=20)
parser.add_argument("--nworkers", type=int, help="number of workers", default=20)
parser.add_argument("--nbyz", type=int, help="number of Byzantine workers", default=2)
parser.add_argument("--byz-type", type=str, help="type of Byzantine workers", choices=['none', 'labelflip', 'signflip'], default='labelflip')
parser.add_argument("--byz-param-a", type=float, help="hyperparameter of Byzantine workers", default=10)
parser.add_argument("--byz-param-b", type=float, help="hyperparameter of Byzantine workers", default=10)
parser.add_argument("--byz-param-c", type=float, help="hyperparameter of Byzantine workers", default=10)
parser.add_argument("--model", type=str, help="model", default='mobilenetv2_1.0')
parser.add_argument("--seed", type=int, help="random seed", default=733)
parser.add_argument("--max-delay", type=int, help="maximum of global delay", default=10)
parser.add_argument("--byz-test", type=str, help="none, kardam, or zeno++", choices=['none', 'kardam', 'zeno++'], default='none')
parser.add_argument("--rho", type=float, help="rho of Zeno++", default=0)
parser.add_argument("--epsilon", type=float, help="epsilon of Zeno++", default=0)
parser.add_argument("--zeno-delay", type=int, help="delay of Zeno++", default=10)
parser.add_argument("--zeno-batchsize", type=int, help="batchsize of Zeno++", default=128)

 
args = parser.parse_args()

# print(args, flush=True)

filehandler = logging.FileHandler(args.log)
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

# set random seed
mx.random.seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# set path
data_dir = os.path.join(args.dir, 'dataset_split_1')
train_dir = os.path.join(data_dir, 'train')
# path to validation data
val_dir = os.path.join(data_dir, 'val')

training_filename = os.path.join(train_dir, 'train_data_000.pkl')
testing_filename = os.path.join(val_dir, 'test_data.pkl')
validation_filename = os.path.join(val_dir, 'val_data.pkl')

context = mx.cpu()

classes = args.classes

# load training data
def load_data(train_filename):
    with open(train_filename, "rb") as f:
        data = pickle.load(f)
        data = pickle.loads(data)
    dataset = mx.gluon.data.dataset.ArrayDataset(data[0], data[1])
    return dataset

def load_model(model_name):
    if model_name == 'default':
        net = gluon.nn.Sequential()
        with net.name_scope():
            #  First convolutional layer
            net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
            net.add(gluon.nn.BatchNorm())
            net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
            net.add(gluon.nn.BatchNorm())
            net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            net.add(gluon.nn.Dropout(rate=0.25))
            #  Second convolutional layer
            # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            # Third convolutional layer
            net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
            net.add(gluon.nn.BatchNorm())
            net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
            net.add(gluon.nn.BatchNorm())
            net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            net.add(gluon.nn.Dropout(rate=0.25))
            # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
            # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
            # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
            # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            # Flatten and apply fullly connected layers
            net.add(gluon.nn.Flatten())
            # net.add(gluon.nn.Dense(512, activation="relu"))
            # net.add(gluon.nn.Dense(512, activation="relu"))
            net.add(gluon.nn.Dense(128, activation="relu"))
            net.add(gluon.nn.Dense(128, activation="relu"))
            net.add(gluon.nn.Dropout(rate=0.25))
            net.add(gluon.nn.Dense(classes))
    else:
        model_kwargs = {'ctx': context, 'pretrained': False, 'classes': classes}
        net = get_model(model_name, **model_kwargs)

    # initialization
    # if model_name.startswith('cifar') or model_name == 'default':
    #     net.initialize(mx.init.Xavier(), ctx=context)
    # else:
    #     net.initialize(mx.init.MSRAPrelu(), ctx=context)
    # net.initialize(mx.init.MSRAPrelu(), ctx=context)
    net.initialize(mx.init.Xavier(), ctx=context)
    return net

model_name = args.model

net = load_model(model_name)

# # no weight decay
# for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
#     v.wd_mult = 0.0

# SGD optimizer
optimizer = 'sgd'
lr = args.lr
optimizer_params = {'momentum': args.momentum, 'learning_rate': lr, 'wd': 0.0001}
# optimizer_params = {'momentum': 0.0, 'learning_rate': lr, 'wd': 0.0}

lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]

loss_func = gluon.loss.SoftmaxCrossEntropyLoss()

train_metric = mx.metric.Accuracy()

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
train_cross_entropy = mx.metric.CrossEntropy()

# training dataset
train_dataset = load_data(training_filename)
train_data = gluon.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, last_batch='rollover', num_workers=0)

# dataset for computing the accuracy on testing dataset
test_test_dataset = load_data(testing_filename)
test_test_data = gluon.data.DataLoader(test_test_dataset, batch_size=1000, shuffle=False, last_batch='keep', num_workers=0)

# dataset for computing the cross-entropy loss on training dataset
test_train_dataset = load_data(training_filename)
test_train_data = gluon.data.DataLoader(test_train_dataset, batch_size=1000, shuffle=False, last_batch='keep', num_workers=0)

# zeno validation, for computing zeno score
val_dataset = load_data(validation_filename)
val_data = gluon.data.DataLoader(val_dataset, batch_size=args.zeno_batchsize, shuffle=True, last_batch='rollover', num_workers=0)
val_data_iter = iter(val_data)

# warmup
print('warm up', flush=True)
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
trainer.set_learning_rate(0.01)

for local_epoch in range(1):
    for i, (data, label) in enumerate(train_data):
        with ag.record():
            outputs = net(data)
            loss = loss_func(outputs, label)
        loss.backward()
        trainer.step(args.batchsize)

nd.waitall()

params_prev = [param.data().copy() for param in net.collect_params().values()]
params_prev_list = [params_prev]

nd.waitall()

if args.byz_test == 'kardam':
    grads_list = []
    lips_list = []
    quantile_q = (args.nworkers-args.b) * 1.0 / args.nworkers
elif args.byz_test == 'zeno++':
    zeno_net = load_model(model_name)
    zeno_trainer = gluon.Trainer(zeno_net.collect_params(), optimizer, optimizer_params)
    zeno_trainer.set_learning_rate(0.001)
    # warm up, mxnet needs running forward/backward for at least once to initizlize the model
    for local_epoch in range(1):
        for i, (data, label) in enumerate(val_data):
            with ag.record():
                outputs = zeno_net(data)
                loss = loss_func(outputs, label)
            loss.backward()
            zeno_trainer.step(args.batchsize)
            break
        break
    nd.waitall()

accept_counter = 0
gradient_counter = 0
false_positive = 0
false_negative = 0
positive = 0.0001
negative = 0.0001


sum_delay = 0
tic = time.time()
# reset optimizer
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
for epoch in range(args.epochs):

    # lr decay
    if epoch in lr_decay_epoch:
        lr = lr * args.lr_decay

    trainer.set_learning_rate(lr)     

    # training
    for i, (data, label) in enumerate(train_data):
        # byzantine
        positive_flag = False
        # kardam requires several iterations without Byzantine failures, in order to initialize the table of "empirical Lipschitz coefficient"
        if args.nbyz > 0 and epoch > 0 and args.byz_type == 'labelflip':
            if random.randint(1, args.nworkers) <= args.nbyz:
                positive_flag = True
                label = label.copy()
                label = args.classes - 1 - label
        # obtain previous model
        if len(params_prev_list)-1 - args.max_delay < 0:
            model_idx = random.randint(0, len(params_prev_list)-1)
        else:
            model_idx = random.randint(len(params_prev_list)-1 - args.max_delay, len(params_prev_list)-1)
        params_prev = params_prev_list[model_idx]
        for param, param_prev in zip(net.collect_params().values(), params_prev):
            if param.grad_req != 'null':
                weight = param.data()
                weight[:] = param_prev
        # compute gradient
        with ag.record():
            outputs = net(data)
            loss = loss_func(outputs, label)
        loss.backward()
        
        # if args.nbyz > 0 and ( epoch > 0 or (epoch == 0 and i >= args.nworkers) ) and args.byz_type == 'signflip':
        if args.nbyz > 0 and epoch > 0 and args.byz_type == 'signflip':
            if random.randint(1, args.nworkers) <= args.nbyz:
                positive_flag = True
                for param in net.collect_params().values():
                    if param.grad_req != 'null':
                        grad = param.grad()
                        grad[:] = - args.byz_param_a * grad

        gradient_counter = gradient_counter + 1
        if args.byz_test == 'kardam':
            byz_flag = True
            lips = 99999
            if len(grads_list) >= args.nworkers:
                accumulate_param = 0
                accumulate_grad = 0
                if model_idx != len(params_prev_list)-1:
                    grads_prev = grads_list[-1]
                    params_prev = params_prev_list[-1]
                else:
                    grads_prev = grads_list[-2]
                    params_prev = params_prev_list[-2]
                for param, param_prev, grad_prev in zip(net.collect_params().values(), params_prev, grads_prev):
                    if param.grad_req != 'null':
                        grad_current = param.grad()
                        param_current = param.data()
                        accumulate_param = accumulate_param + nd.square(param_current - param_prev).sum()
                        accumulate_grad = accumulate_grad + nd.square(grad_current - grad_prev).sum()
                lips = math.sqrt(accumulate_grad.asscalar()) / math.sqrt(accumulate_param.asscalar())
                if lips <= np.quantile(lips_list, quantile_q):
                    byz_flag = False
                    accept_counter = accept_counter + 1
                nd.waitall()
            else:
                byz_flag = False
                accept_counter = accept_counter + 1
        elif args.byz_test == 'zeno++':
            zeno_max_delay = args.zeno_delay
            zeno_rho = args.rho
            zeno_epsilon = args.epsilon
            byz_flag = True
            if i % zeno_max_delay == 0:
                # obtain previous model
                model_idx = len(params_prev_list)-1
                params_prev = params_prev_list[model_idx]
                for param, param_prev in zip(zeno_net.collect_params().values(), params_prev):
                    if param.grad_req != 'null':
                        weight = param.data()
                        weight[:] = param_prev
                # compute g_r
                zeno_trainer = gluon.Trainer(zeno_net.collect_params(), optimizer, optimizer_params)
                zeno_trainer.set_learning_rate(lr) 
                data_pair = next(val_data_iter, None)
                if data_pair is None:
                    val_data_iter = iter(val_data)
                    data_pair = next(val_data_iter, None)
                with ag.record():
                    outputs = zeno_net(data_pair[0])
                    loss = loss_func(outputs, data_pair[1])
                loss.backward()
                nd.waitall()
            # normalize g
            param_square = 0
            zeno_param_square = 0
            for param, zeno_param in zip(net.collect_params().values(), zeno_net.collect_params().values()):
                if param.grad_req != 'null':
                    param_square = param_square + param.grad().square().sum()
                    zeno_param_square = zeno_param_square + zeno_param.grad().square().sum()
            c = math.sqrt( zeno_param_square.asscalar() / param_square.asscalar() )
            for param in net.collect_params().values():
                if param.grad_req != 'null':
                    grad = param.grad()
                    grad[:] = grad * c
            # compute zeno score
            zeno_innerprod = 0
            zeno_square = param_square
            for param, zeno_param in zip(net.collect_params().values(), zeno_net.collect_params().values()):
                if param.grad_req != 'null':
                    zeno_innerprod = zeno_innerprod + nd.sum(param.grad() * zeno_param.grad())
            score = args.lr * (zeno_innerprod.asscalar()) - zeno_rho * (zeno_square.asscalar()) + args.lr * zeno_epsilon
            if score >= 0:
                byz_flag = False
                accept_counter = accept_counter + 1
            nd.waitall()

        else:
            byz_flag = False
            accept_counter = accept_counter + 1

        if positive_flag == True:
            positive = positive + 1
        else:
            negative = negative + 1
        
        if positive_flag == False and byz_flag == True:
            false_positive = false_positive + 1
        if positive_flag == True and byz_flag == False:
            false_negative = false_negative + 1
            
        # bring back the current model
        params_prev = params_prev_list[-1]
        for param, param_prev in zip(net.collect_params().values(), params_prev):
            if param.grad_req != 'null':
                weight = param.data()
                weight[:] = param_prev

        # byz test
        if byz_flag == False:
            # update
            trainer.step(args.batchsize)

            nd.waitall()

            # save model to queue
            params_prev_list.append([param.data().copy() for param in net.collect_params().values()])
            if len(params_prev_list) > args.nworkers * 2:
                del params_prev_list[0]

            if args.byz_test == 'kardam':
                # update the list of gradients and lips constant
                grads_list.append([param.grad().copy() if param.grad_req != 'null' else None for param in net.collect_params().values()])
                lips_list.append(lips)
                if len(grads_list) > max(args.nworkers * 2, args.max_delay):
                    del grads_list[0]
                    del lips_list[0]
        
            nd.waitall()

    
    # validation
    if  epoch % args.interval == 0 or epoch == args.epochs-1:
        acc_top1.reset()
        acc_top5.reset()
        train_cross_entropy.reset()
        # get accuracy on testing data
        for i, (data, label) in enumerate(test_test_data):
            outputs = net(data)
            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)

        # get cross entropy loss on traininig data
        for i, (data, label) in enumerate(test_train_data):
            outputs = net(data)
            train_cross_entropy.update(label, nd.softmax(outputs))

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        _, crossentropy = train_cross_entropy.get()

        logger.info('[Epoch %d] validation: acc-top1=%f acc-top5=%f, loss=%f, fp=%f, fn=%f, lr=%f, accept ratio=%f, max_delay=%d, time=%f' % (epoch, top1, top5, crossentropy, false_positive/negative, false_negative/positive, trainer.learning_rate, accept_counter/gradient_counter, args.max_delay, time.time()-tic))
        tic = time.time()
        
        nd.waitall()



            