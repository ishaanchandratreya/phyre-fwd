#!/bin/bash

session1="00020"
session2="00017"
session3="00018"

tmux new-session -d -s $session1
tmux send-keys -t $session1 'sgpu 5' C-m
tmux send-keys -t $session1 'afwdm' C-m 'cd ../world_models/PlaNet' C-m "python main_phyre.py --id phyre_${session1} --env ${session1}" C-m

tmux new-session -d -s $session2
tmux send-keys -t $session2 'sgpu 6' C-m
tmux send-keys -t $session2 'afwdm' C-m 'cd ../world_models/PlaNet' C-m "python main_phyre.py --id phyre_${session2} --env ${session2}" C-m

tmux new-session -d -s $session3
tmux send-keys -t $session3 'sgpu 7' C-m
tmux send-keys -t $session3 'afwdm' C-m 'cd ../world_models/PlaNet' C-m "python main_phyre.py --id phyre_${session3} --env ${session3}" C-m