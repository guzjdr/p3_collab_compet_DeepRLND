[//]: # (Image References)
[image1]: data_files/Tennis_Court.png 

# Project3: Collaboration and Competition ( Deep RL NanoDegree )

This repository contains material related to Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program. 

## Project Description

Train a TD3 Agent(s) to learn an optimal policy that maximizes the expected return ( score ) while playing table tennis. 


## Environment


Unity Tennis Environment - Used to train an agent to solve the task described above. The main idea is to train a set of agents to learn to play table tennis within this environment. 

<b>**Unity Tennis Environment**</cebnter>

![alt text][image1] 





## Getting Started 

Note the following conda environment was created on a Linux machine ( __Linux Ubuntu 20.04.6 LTS__ ) 

1. Please clone the following Udacity Repo: [deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning) Repo.

2. Follow the instructions - inside the repo - to set up the necessary dependencies. 
* Create (and activate) a new conda environment with Python 3.6 : 
```bash
conda create --name drlnd python=3.6
source activate drlnd
```
* Install OpenAI gym
```bash
pip install gym
```
* Install the necessary Python dependencies 
```bash
cd deep-reinforcement-learning/python
```
- Modified Requirements.txt 
```text
Pillow>=4.2.1
matplotlib
numpy>=1.11.0
jupyter
pytest>=3.2.2
docopt
pyyaml
protobuf==3.5.2
grpcio==1.11.0
torch==1.7.0
pandas
scipy
ipykernel
```
```bash
pip install .
```
* Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment. 
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
* Download the Unity Environment(s):
Please refer to this repo to download the environment files: [p3_collab-compet](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet)
