#!/bin/bash

cd ..
python visualizer.py --models "/proj/vondrick/ishaan/phyre-fwd/world_models/PlaNet/results/phyre_00000_20/models_200.pth" --stride 20 --env "00000"
python visualizer.py --models "/proj/vondrick/ishaan/phyre-fwd/world_models/PlaNet/results/phyre_00000_40/models_200.pth" --stride 40 --env "00000"
python visualizer.py --models "/proj/vondrick/ishaan/phyre-fwd/world_models/PlaNet/results/phyre_00000_5/models_200.pth" --stride 5 --env "00000"
python visualizer.py --models "/proj/vondrick/ishaan/phyre-fwd/world_models/PlaNet/results/phyre_00011/models_200.pth" --stride 60 --env "00011"
python visualizer.py --models "/proj/vondrick/ishaan/phyre-fwd/world_models/PlaNet/results/phyre_00017/models_200.pth" --stride 60 --env "00017"
python visualizer.py --models "/proj/vondrick/ishaan/phyre-fwd/world_models/PlaNet/results/phyre_00018/models_200.pth" --stride 60 --env "00018"
python visualizer.py --models "/proj/vondrick/ishaan/phyre-fwd/world_models/PlaNet/results/phyre_00020/models_200.pth" --stride 60 --env "00020"
python visualizer.py --models "/proj/vondrick/ishaan/phyre-fwd/world_models/PlaNet/results/phyre_00020_10/models_200.pth" --stride 10 --env "00020"
python visualizer.py --models "/proj/vondrick/ishaan/phyre-fwd/world_models/PlaNet/results/phyre_00020_20/models_200.pth" --stride 20 --env "00020"
python visualizer.py --models "/proj/vondrick/ishaan/phyre-fwd/world_models/PlaNet/results/phyre_00020_5/models_200.pth" --stride 5 --env "00020"
python visualizer.py --models "/proj/vondrick/ishaan/phyre-fwd/world_models/PlaNet/results/phyre_00000:000/models_200.pth" --stride 60 --env "00000:000"