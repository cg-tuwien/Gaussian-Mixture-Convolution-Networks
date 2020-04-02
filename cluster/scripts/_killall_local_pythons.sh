#!/bin/bash
for pid in $(top -bn1 | grep python | grep acelarek | grep -oE "[0-9]+.+" | grep -oE "^[0-9]+")
do
	kill -sigterm $pid
done 
