#!/bin/bash
#PBS -l select=11:ncpus=112 -lplace=excl

source /opt/intel/compilers_and_libraries_2017/linux/mpi/bin64/mpivars.sh
source activate mxnet_latest


### OPA FABRIC ###
export I_MPI_FABRICS=ofi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=56


### OPA FABRIC ###
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

export KMP_AFFINITY=granularity=fine,compact,1,0;

basedir=/home/user/zeno_async
logdir=$basedir/results

# training data
inputdir=$basedir/tmp

watchfile=$logdir/experiment_script_2.log

# prepare the training dataset
# python convert_cifar10.py --nsplit 1 --output $inputdir

model="default"
lr=0.1
method="zeno++"
byz="signflip"
maxdelay=10
nbyz=6
rho=0.001
epsilon=0
zenodelay=10
seed=337

logfile=$logdir/zeno_async_serveronly.txt
> $logfile

cd /homes/cx2/src/byz/zeno_async

for seed in 337 733 773 377 112 211 557 755 577 775;
do
    python train_cifar10_server.py --classes 10 --model ${model} --nworkers 10 --nbyz ${nbyz} --byz-type ${byz} --byz-test ${method} --rho ${rho} --epsilon ${epsilon} --zeno-delay ${zenodelay} --batchsize 128 --lr ${lr} --lr-decay 0.1 --lr-decay-epoch 100,150 --epochs 200 --seed ${seed} --max-delay ${maxdelay} --dir $inputdir --log $logfile 2>&1 | tee $watchfile
done
