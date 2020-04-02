#!/bin/bash
echo "executing \"$1\""
for i in {1..5}; do
    echo -e "\e[7m\e[93mmdl${i}\e[27m\e[39m"
    ssh dl$i "$1"
done
