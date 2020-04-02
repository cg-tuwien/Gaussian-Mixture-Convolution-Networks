#!/bin/bash
_killall_dl_tmuxes.sh
sleep 5
_execute_command.sh "rm -rf data/tensorboard"
_start_all_learning.sh
