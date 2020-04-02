#!/bin/bash
echo "checking load"
for i in {1..5}; do
    echo -e "\e[7m\e[93mmdl${i}\e[27m\e[39m"
    ssh dl$i "top -bn 1 | head -n 12; echo \"====\"; nvidia-smi"
    echo -e "\n\n"
done
