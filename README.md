[![GitHub license](https://img.shields.io/badge/license-Apache-blue.svg)](LICENSE)
# Goals

The goal of this version of the library is threefold:
1. To expand capacities of the excellent PHYRE simulation from FAIR to possibly work better with their own
algorithms to solve the task
2. To make the PHYRE task amenable to solving using some other interesting families of models (eg. world models)
3. To make the PHYRE project generally more interpretable to new developers who wish to taylor it to their needs/tasks. Principally, I hope
that some of the great flexibilites afforded by the Box2D library are made more accessible through the Python interface, and the Creator API
   is expanded to make for more interesting tasks without changing too much of the original structure.

#Background 

This repository is built on top of the PHYRE implementation in PHYRE Forward Agents, which is the code for reproducing [Forward Prediction for Physical Reasoning](https://arxiv.org/abs/2006.10734). 
It further contains code from [an open source PyTorch PlaNet implementation](https://github.com/Kaixhin/PlaNet). The PlaNet World Model was developed at Google AI by [@danijar](https://github.com/danijar) and team, although the
particular repository from which I modify the implementation is by [@Kaixhin](https://github.com/Kaixhin).

Added elements to PHYRE simulation include:

1. A **full** gym-like API for framing the PHYRE task in a continuous control manner. This is built by staying as close
to the original coding abstractions in the PHYRE repository, namely maintaining the interface with FeaturizedObjects, 
   ActionMapper and other classes. Modifications made here extend all the way through the bindings to the C++ simulation files. 
   A complete documentation is being prepared for the modifications made.
   
2. More **control** over PHYRE, including scene features in the form of object velocity and angular velocity for all available objects, and the ability to set these values 
 in the action mappers and the creator API. This enables the step-wise execution of PHYRE, and the ability to have objects 
   moving at timestep 0. This also enables a richer action space in the task, instead of only three dimensions. I have plans to extend this 
   further to allow ease of control over shapes, angles, etc. 
   
3. A plug-in under /world-models to integrate the PHYRE task (now as a continuous control task, with constraints) into world models for 
planning. Currently, we have implemented PlaNet without reward. Without a proper reward signal, this will learn the deterministic/stochastic
   pathways for transition in the model, however, there is no exploration signal. There is plans to expand the set of available models to include
   those trained with curiosity etc. Further, I have provided a Visualizer class to see the belief/posterior visualizations
   
# Modifications (and making more)

**src/if** contains Apache Thrift files that support cross platform development and define data type that
are used both in Python and C++. Structs such as Body, CircleWithPosition etc have been expanded here to further 
hold velocity and angular velocity values

Modifications needed to be made to the following files to populate the relevant instance fields added
to the structs from the Box2D simulation and report it in the featurization of the different objects.

**src/simulator** contains the 2D C++ simulation that is built on top of Box2D library. 

- **simulation_bindings.cpp** contains C++ functions for a range of functions called inside the python code. In particular
the magic_ponies function is critical to simulation execution.
  
- **image_to_box2d.cpp** and **image_to_box2d.h** controls encoding any input objects, etc into our data structs and extract informatioon
  from this structs to return to the Python code as simulation features. Functions include
mergeUserInputIntoScene, featurizeBody, featurizeScene
  
- **task_utils.cpp** and **task_utils.h**  contains overall abstraction of simulating task including 
  making perturbations, instantiating the Box2D world, setting parameters, running the sim steps in Box2D, and so on.
  
- **thrift_box2d_conversion.cpp** and **thrift_box2d_conversion.h** is the main interface converting the Thrift abstractions 
  that we have defined into equivalent Box2D code (eg b2BodyDef). Functions contained include 
  convertSceneToBox2dWorld, addBodiesToWorld, updateSceneFromWorld.

- **creator.cpp** and **creator.h** contains functions for the creator API, and the ability to set paraameters of the different 
structs in a conveninent iterative way, as defined in the original documentation. This was extended to the new parameters,
  including velocity and angular velocity. 

- **task_utils_parallel.cpp** allows parallel simulation of tasks.

- **task_validation.cpp** and **task_validation.h** contains a bunch of low level math functions and verifies if there are collisions etc 

- **task_io.cpp** and **task_io.h** contains miscellaneous logging for tasks.

- **tests** needed to be modified since the Thrift files eg. updates had changed. 


- **simulator.py** contains main code for calling underlying simulation, including magic ponies.

- **simulation.py** keeps track of the high level featurized objects class defined in python. This has 
been significant modified for to contain more info re: the simulation and facilitate the Gym implementation

- **action_mappers.py** handles the relevant scaling from [0, 1] interface available to users to the actual units associated
with the scene/velocity etc. I've added several new action mappers to work with velocity, etc.

- **action_simulator.py** is a wrapper around simulator with some user facing capabilies. I added the option
to initialize pre-compiled simulation which is important to the gym interface I built
  
**agents** contains mostly the algorithms trained by the original authors (Giridhar et. al) for the task.
To this I added the main file

- **phyre_gym.py** This contains the entire Gym interface that allows for two modes
    1. Physics: you take an action and watch it play out, resetting at the end
    2. CC (Continuous Control): you are allowed to take a new action at each time step under the following model
    
Receive ACTION -> run 1 stride of simulation -> develop and instantiate new task from the status after 1 stride -> take new action. 







# Getting started

This section follows the installation instructions from the original paper. 

## Installation
A [Conda](https://docs.conda.io/en/latest/) virtual enviroment is provided contianing all necessary dependencies.
```(bash)
git clone https://github.com/facebookresearch/phyre-fwd.git
cd phyre-fwd
conda env create -f env.yml && conda activate phyre-fwd
pushd src/python && pip install -e . && popd
pip install -r requirements.agents.txt
```
Make a directory somewhere with enough space for trained models, and symlink it to `agents/outputs`.

## Methods

For training the methods, and available pre-trained models, see [agents](agents/).

# License
PHYRE forward agents are released under the Apache license. See [LICENSE](LICENSE) for additional details.


# Citation

If you use `phyre-fwd` or the baseline results, please cite it:

```bibtex
@article{girdhar2020forward,
    title={Forward Prediction for Physical Reasoning},
    author={Rohit Girdhar and Laura Gustafson and Aaron Adcock and Laurens van der Maaten},
    year={2020},
    journal={arXiv:2006.10734}
}
```
