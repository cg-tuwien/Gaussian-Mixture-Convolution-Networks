#!/bin/bash
tmux new -d "tensorboard --logdir=$HOME/data/tensorboard --port=10080 --bind_all"
tmux new -d "tensorboard --logdir=$HOME/data/backup_20200401_combined_vs_alternate/tensorboard --port=10081 --bind_all"
tmux new -d "tensorboard --logdir=$HOME/data/backup_20200329_1740_splitGrad/tensorboard --port=10082 --bind_all"
tmux new -d "tensorboard --logdir=$HOME/data/backup_20200326_1620_combined_learning50_smallFittingNets/tensorboard --port=10083 --bind_all"
tmux new -d "tensorboard --logdir=$HOME/data/backup_20200325_1250_combined_learning/tensorboard --port=10084 --bind_all"
tmux new -d "tensorboard --logdir=$HOME/data/backup_20200322_1800_subdivider_nets/tensorboard --port=10085 --bind_all"

