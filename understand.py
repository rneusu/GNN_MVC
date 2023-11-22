import sys
import os
from discrete_actor_critic import DiscreteActorCritic
from MVC import MVC
import matplotlib.pyplot as plt
from smooth_signal import smooth
import numpy as np
import time
import torch

sys.stdout = open('output/understand.txt','wt')

cuda_flag = False
num_nodes = 30
p_edge = 0.15
mvc = MVC(num_nodes,p_edge)
ndim = mvc.get_graph_dims()

init_state,done = mvc.reset()


