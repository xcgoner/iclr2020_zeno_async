import argparse, time, logging, os, math, random
os.environ["MXNET_USE_OPERATOR_TUNING"] = "0"


import numpy as np
from scipy import stats
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms


import gluonnlp as nlp

from os import listdir
import os.path
import argparse

import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--batchsize", type=int, help="batchsize", default=20)
parser.add_argument("--epochs", type=int, help="number of epochs", default=100)
parser.add_argument("--interval", type=int, help="log interval (epochs)", default=10)
parser.add_argument("--lr", type=float, help="learning rate", default=20)
parser.add_argument("--lr-decay", type=float, help="lr decay rate", default=0.5)
parser.add_argument("--lr-decay-epoch", type=str, help="lr decay epoch", default='2000')
parser.add_argument("--momentum", type=float, help="momentum", default=0)
parser.add_argument("--log", type=str, help="dir of the log file", default='train_wikitext.log')
parser.add_argument("--nworkers", type=int, help="number of workers", default=20)
parser.add_argument("--nbyz", type=int, help="number of Byzantine workers", default=2)
parser.add_argument("--byz-type", type=str, help="type of Byzantine workers", choices=['none', 'signflip'], default='signflip')
parser.add_argument("--byz-param-a", type=float, help="hyperparameter of Byzantine workers", default=10)
parser.add_argument("--byz-param-b", type=float, help="hyperparameter of Byzantine workers", default=10)
parser.add_argument("--byz-param-c", type=float, help="hyperparameter of Byzantine workers", default=10)
parser.add_argument("--model", type=str, help="model", default='standard_lstm_lm_200')
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

logger.info(args)

# set random seed
mx.random.seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

bptt = 35
grad_clip = 0.25
batch_size = args.batchsize

# Load the dataset
train_dataset, val_dataset, test_dataset = [
    nlp.data.WikiText2(
        segment=segment, bos=None, eos='<eos>', skip_empty=False)
    for segment in ['train', 'val', 'test']
]

# Extract the vocabulary and numericalize with "Counter"
vocab = nlp.Vocab(
    nlp.data.Counter(train_dataset), padding_token=None, bos_token=None)

# Batchify for BPTT
bptt_batchify = nlp.data.batchify.CorpusBPTTBatchify(
    vocab, bptt, batch_size, last_batch='discard')
train_data, val_data, test_data = [
    bptt_batchify(x) for x in [train_dataset, val_dataset, test_dataset]
]

context = [mx.cpu()]

model_name = args.model

net, vocab = nlp.model.get_model(model_name, vocab=vocab, dataset_name=None)

# initialization
net.initialize(mx.init.Xavier(), ctx=context)

# # no weight decay
# for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
#     v.wd_mult = 0.0

# SGD optimizer
optimizer = 'sgd'
lr = args.lr
optimizer_params = {'momentum': args.momentum, 'learning_rate': lr, 'wd': 0}
# optimizer_params = {'momentum': 0.0, 'learning_rate': lr, 'wd': 0.0}

lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]

loss_func = gluon.loss.SoftmaxCrossEntropyLoss()

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(i) for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


# Note that ctx is short for context
def evaluate(model, data_source, batch_size, ctx):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(
        batch_size=batch_size, func=mx.nd.zeros, ctx=ctx)
    for i, (data, target) in enumerate(data_source):
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        output, hidden = model(data, hidden)
        hidden = detach(hidden)
        L = loss_func(output.reshape(-3, -1), target.reshape(-1))
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal

nworkers = args.nworkers
train_data_list = []
for i, (data, target) in enumerate(train_data):
    data_list = gluon.utils.split_and_load(data, context,
                                                batch_axis=1, even_split=True)
    target_list = gluon.utils.split_and_load(target, context,
                                                batch_axis=1, even_split=True)
    train_data_list.append([data_list, target_list])

# zeno validation, for computing zeno score
val_data_list = []
for i, (data, target) in enumerate(val_data):
    data_list = gluon.utils.split_and_load(data, context,
                                                batch_axis=1, even_split=True)
    target_list = gluon.utils.split_and_load(target, context,
                                                batch_axis=1, even_split=True)
    val_data_list.append([data_list, target_list])

print('data cached')

parameters = net.collect_params().values()

# warmup
print('warm up', flush=True)
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
trainer.set_learning_rate(1)
# train    
hiddens = [net.begin_state(batch_size//len(context), func=mx.nd.zeros, ctx=ctx)
            for ctx in context]
random.shuffle(train_data_list)
for i, data in enumerate(train_data_list):
    data_list = data[0]
    target_list = data[1]
    hiddens = detach(hiddens)
    L = 0
    Ls = []

    with autograd.record():
        for j, (X, y, h) in enumerate(zip(data_list, target_list, hiddens)):
            output, h = net(X, h)
            batch_L = loss_func(output.reshape(-3, -1), y.reshape(-1,))
            L = L + batch_L.as_in_context(context[0]) / (len(context) * X.size)
            Ls.append(batch_L / (len(context) * X.size))
            hiddens[j] = h
    L.backward()
    grads = [p.grad(x.context) for p in parameters for x in data_list]
    gluon.utils.clip_global_norm(grads, grad_clip)

    trainer.step(1)

nd.waitall()

params_prev = [param.data().copy() for param in parameters]
params_prev_list = [params_prev]

nd.waitall()

if args.byz_test == 'kardam':
    grads_list = []
    lips_list = []
    quantile_q = (args.nworkers-args.b) * 1.0 / args.nworkers
elif args.byz_test == 'zeno++':
    zeno_net, _ = nlp.model.get_model(model_name, vocab=vocab, dataset_name=None)
    zeno_net.initialize(mx.init.Xavier(), ctx=context)
    zeno_trainer = gluon.Trainer(zeno_net.collect_params(), optimizer, optimizer_params)
    zeno_trainer.set_learning_rate(0.001)
    # warm up, mxnet needs running forward/backward for at least once to initizlize the model
    hiddens = [zeno_net.begin_state(batch_size//len(context), func=mx.nd.zeros, ctx=ctx)
                for ctx in context]
    random.shuffle(train_data_list)
    for i, data in enumerate(train_data_list):
        data_list = data[0]
        target_list = data[1]
        hiddens = detach(hiddens)
        L = 0
        Ls = []

        with autograd.record():
            for j, (X, y, h) in enumerate(zip(data_list, target_list, hiddens)):
                output, h = zeno_net(X, h)
                batch_L = loss_func(output.reshape(-3, -1), y.reshape(-1,))
                L = L + batch_L.as_in_context(context[0]) / (len(context) * X.size)
                Ls.append(batch_L / (len(context) * X.size))
                hiddens[j] = h
        L.backward()
        grads = [p.grad(x.context) for p in parameters for x in data_list]
        gluon.utils.clip_global_norm(grads, grad_clip)
        trainer.step(1)
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
best_val = float("Inf")
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
for epoch in range(args.epochs):

    # # lr decay
    # if epoch in lr_decay_epoch:
    #     lr = lr * args.lr_decay

    trainer.set_learning_rate(lr)     

    # training
    hiddens = [net.begin_state(batch_size//len(context), func=mx.nd.zeros, ctx=ctx)
                    for ctx in context]
    for i, data in enumerate(train_data_list):
        data_list = data[0]
        target_list = data[1]
        hiddens = detach(hiddens)
        L = 0
        Ls = []
        # byzantine
        positive_flag = False
        # kardam requires several iterations without Byzantine failures, in order to initialize the table of "empirical Lipschitz coefficient"
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
        with autograd.record():
            for j, (X, y, h) in enumerate(zip(data_list, target_list, hiddens)):
                output, h = net(X, h)
                batch_L = loss_func(output.reshape(-3, -1), y.reshape(-1,))
                L = L + batch_L.as_in_context(context[0]) / (len(context) * X.size)
                Ls.append(batch_L / (len(context) * X.size))
                hiddens[j] = h
        L.backward()
        grads = [p.grad(x.context) for p in parameters for x in data_list]
        gluon.utils.clip_global_norm(grads, grad_clip)
        
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
                val_data_pair = random.choice(val_data_list)
                data_list = val_data_pair[0]
                target_list = val_data_pair[1]
                hiddens = detach(hiddens)
                with autograd.record():
                    for j, (X, y, h) in enumerate(zip(data_list, target_list, hiddens)):
                        output, h = zeno_net(X, h)
                        batch_L = loss_func(output.reshape(-3, -1), y.reshape(-1,))
                        L = L + batch_L.as_in_context(context[0]) / (len(context) * X.size)
                        Ls.append(batch_L / (len(context) * X.size))
                        hiddens[j] = h
                L.backward()
                grads = [p.grad(x.context) for p in parameters for x in data_list]
                gluon.utils.clip_global_norm(grads, grad_clip)
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
            trainer.step(1)

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
    if  epoch % args.interval == 0 or epoch == args.iterations-1:
        val_L = evaluate(net, test_data, batch_size, context[0])

        logger.info('[Epoch %d] test: loss=%f, ppl=%f, fp=%f, fn=%f, lr=%f, time=%f' % (epoch, val_L, math.exp(val_L), false_positive/negative, false_negative/positive, trainer.learning_rate, time.time()-tic))
        tic = time.time()
        
        nd.waitall()

        if val_L < best_val:
            best_val = val_L
        else:
            lr *= 0.25



            