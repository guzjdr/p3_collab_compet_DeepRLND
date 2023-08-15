# Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Python Modules
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import pdb;

#Unity Environment
from unityagents import UnityEnvironment

#Torch GPU configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")