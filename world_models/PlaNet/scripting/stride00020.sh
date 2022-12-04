#!/bin/bash

session1='stride5'
session2='stride10'
session3='stride20'

tmux new-session -d -s $session1
tmux send-keys -t $session1 'sgpu 0' C-m
tmux send-keys -t $session1 'afwdm' C-m 'cd ../world_models/PlaNet' C-m "python main_phyre.py --id phyre_00020_5 --env 00020 --stride 5" C-m

tmux new-session -d -s $session2
tmux send-keys -t $session2 'sgpu 1' C-m
tmux send-keys -t $session2 'afwdm' C-m 'cd ../world_models/PlaNet' C-m "python main_phyre.py --id phyre_00020_10 --env 00020 --stride 10" C-m

tmux new-session -d -s $session3
tmux send-keys -t $session3 'sgpu 2' C-m
tmux send-keys -t $session3 'afwdm' C-m 'cd ../world_models/PlaNet' C-m "python main_phyre.py --id phyre_00020_20 --env 00020 --stride 20" C-m