#!/bin/bash
echo "starting learning"

# starting python in unbuffered mode (-u) in order to be able to monitor the logfile

mkdir -p data/logs
mkdir -p data/weights

experiment_list=({12..15})
# experiment_list=(0 2 4 5 10 11)
nodes="dl4"
experiment_index=0

for dl in $nodes; do
    echo -e "\e[7m\e[93m===${dl}===\e[27m\e[39m"
    for gpu in {0..3}; do
        experiment=${experiment_list[$experiment_index]}
        echo "starting experiment $experiment on $dl on gpu $gpu"
        ssh $dl "tmux new -d \"python -uO gmc_net/experiment_dl_$experiment.py cuda:$gpu |& tee data/logs/experiment_dl_$experiment.log\""
        experiment_index=$((experiment_index + 1))
        if [ $experiment_index -ge ${#experiment_list[@]} ]; then
            echo "all experiments started"
            exit 0
        fi
    done
done
echo "not everything started, ran out of nodes"

# 
# dl="dl1"
# echo -e "\e[7m\e[93mm${dl}\e[27m\e[39m"
# for i in {0..3}; do
#     echo "starting experiment $i on $dl"
#     ssh $dl "tmux new -d \"python -uO gmc_net/experiment_dl_$i.py |& tee data/logs/experiment_dl_$i.log\""
# done
# 
# dl="dl3"
# echo -e "\e[7m\e[93mm${dl}\e[27m\e[39m"
# for i in {4..7}; do
#     echo "starting experiment $i on $dl"
#     ssh $dl "tmux new -d \"python -uO gmc_net/experiment_dl_$i.py |& tee data/logs/experiment_dl_$i.log\""
# done
# 
# dl="dl4"
# echo -e "\e[7m\e[93mm${dl}\e[27m\e[39m"
# for i in {8..11}; do
#     echo "starting experiment $i on $dl"
#     ssh $dl "tmux new -d \"python -uO gmc_net/experiment_dl_$i.py |& tee data/logs/experiment_dl_$i.log\""
# done
