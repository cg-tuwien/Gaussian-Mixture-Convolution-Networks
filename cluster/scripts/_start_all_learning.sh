#!/bin/bash
echo "starting learning"

mkdir -p data/logs
mkdir -p data/weights

dl="dl1"
echo -e "\e[7m\e[93mm${dl}\e[27m\e[39m"
for i in {0..3}; do
    echo "starting experiment $i on $dl"
    ssh $dl "tmux new -d \"python -O gmc_net/experiment_dl_$i.py |& tee data/logs/experiment_dl_$i.log\""
done

dl="dl3"
echo -e "\e[7m\e[93mm${dl}\e[27m\e[39m"
for i in {4..7}; do
    echo "starting experiment $i on $dl"
    ssh $dl "tmux new -d \"python -O gmc_net/experiment_dl_$i.py |& tee data/logs/experiment_dl_$i.log\""
done

dl="dl4"
echo -e "\e[7m\e[93mm${dl}\e[27m\e[39m"
for i in {8..11}; do
    echo "starting experiment $i on $dl"
    ssh $dl "tmux new -d \"python -O gmc_net/experiment_dl_$i.py |& tee data/logs/experiment_dl_$i.log\""
done
