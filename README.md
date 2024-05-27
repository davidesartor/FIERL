# FIERL (PyTorch implementation)
This code implements 'Fault Identification Enhacement with Reinforcement Learning (FIERL)', an approach to perform active fault diagnosis with reinforcement learning. 

It contains the implemenentation of a model-based fault observer with a Kalman-filter derivation and uses CPO algorithm (Constrained Policy Optimization) to find an integrated control input to maximize the observer performance while ensuring a user-defined performance in tracking control 

The environment accepts any system of the class `FaultyActuatorNoisySystem`. In our experiments, we used the Three-Tank benchmark. 
The observer an updated version. 

## Installation
To install: 
```
git clone https://github.com/davidesartor/FIERL.git
cd FIERL
pip install -e .
```

## Usage 
**Train an agent**: to train CPO agent using Three-Tank system: 
```
python -m experiments.train_3tank 
```

**Watch trained policy**
```
python -m experiments.evaluate_3tank
```