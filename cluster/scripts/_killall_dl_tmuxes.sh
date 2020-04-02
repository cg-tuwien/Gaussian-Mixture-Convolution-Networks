#!/bin/bash
echo "killing tmuxes"
for i in {1..5}; do
    echo -e "\e[7m\e[93mmdl${i}\e[27m\e[39m"
    ssh dl$i _killall_local_tmuxes.sh
done
