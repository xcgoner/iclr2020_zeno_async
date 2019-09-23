# Zeno

### This is the python implementation of the paper "Zeno++: Robust Asynchronous SGD with Arbitrary Number of Byzantine Workers"

### Requirements

The following python packages needs to be installed by pip:

1. MXNET (we use GPU, thus mxnet-cu80 is preferred)
2. Gluon-CV
3. Numpy

The users can simply run the following commond in their own virtualenv:

```bash
pip install --no-cache-dir numpy mxnet-mkl gluoncv
```

### Run the demo

#### Options:

| Option     | Desctiption | 
| ---------- | ----------- | 
|--dir| path of datasets|
|--batch_size 128| batch size of the workers|
|--nepochs 200| total number of epochs|
|--interval 10| log interval|
|--lr 0.1| learning rate|
|--lr-decay 0.1| rate of diminishing learning rate|
|--lr-decay-epoch 100,150| epochs where the learning rate decays|
|--classes | number of classes|
|--nworkers 20| number of workers|
|--nbyz | number of faulty workers|
|--byz_type | type of failures, signflip or labelflip|
|--byz-param-a | hyperparameter of Byzantine workers|
|--byz-param-b | hyperparameter of Byzantine workers|
|--byz-param-c | hyperparameter of Byzantine workers|
|--model | name of neural network|
|--seed 337 | random seed|
|--max-delay 15 | maximum of global delay|
|--byz-test | Byzantine tolerant algorithms: none, kardam, or zeno++|
|--rho | hyperparameter \rho of Zeno++|
|--epsilon | hyperparameter \epsilon of Zeno++|
|--zeno-delay 10 | delay of g_r in Zeno++|
|--zeno-batchsize 10 | batchsize of  Zeno++, n_s in the paper|


* Train with 10 workers, 6 of them are faulty with bit-flipping failures, Zeno as aggregation:
```bash
python train_cifar10.py --classes 10 --model default --nworkers 10 --nbyz 6 --byz-type signflip --byz-test zeno++--rho 0.001 --epsilon 0 --zeno-delay 10 --batchsize 128 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --epochs 200 --seed 337 --max-delay 10 --dir $inputdir --log $logfile 2>&1 | tee $watchfile
```


More detailed commands/instructions can be found in the demo script *experiment_script_1.sh*

