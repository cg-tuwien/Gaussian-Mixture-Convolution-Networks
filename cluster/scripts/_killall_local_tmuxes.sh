#!/bin/bash
for pid in $(top -bn1 | grep tmux | grep acelarek | grep -oE "[0-9]+.+" | grep -oE "^[0-9]+")
do
	kill -sigterm $pid
done 
