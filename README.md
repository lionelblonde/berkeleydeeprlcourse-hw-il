# Imitation learning warm-up

This repository contains the material for the first homework of the deep reinforcement 
learning course taught at Berkeley.
This first homework is on **imitation learning**.
The objective is to implement, on top of the tools initially provided in the repository, 
the Behavioral Cloning algorithm.

Here are the indications given in the original repository (`berkeleycourse/homework/hw1`):

Dependencies: TensorFlow, MuJoCo version 1.31, OpenAI Gym

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible 
with recent Mac machines. There is a request for OpenAI to support it that can be followed 
[here](https://github.com/openai/gym/issues/638).

The only file that you need to look at is `run_expert.py`, which loads up an expert policy, 
runs a specified number of roll-outs, and saves the collected trajectories to disk.

In `experts/`, the provided expert policies are:

* Ant-v1.pkl
* HalfCheetah-v1.pkl
* Hopper-v1.pkl
* Humanoid-v1.pkl
* Reacher-v1.pkl
* Walker2d-v1.pkl

The name of the pickle file corresponds to the name of the gym environment.
