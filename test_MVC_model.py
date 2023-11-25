# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 07:57:52 2019

@author: Or
"""

import torch
from MVC import MVC
import dgl
import torch.nn.functional as F
from Models import ACNet
import time
from copy import deepcopy as dc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from local_search import LOCALSEARCH
from GTL import GTL


cuda_flag = False
num_nodes = 30
p_edge = 0.05

def f():
    mvc = MVC(num_nodes,p_edge)
    ndim = mvc.get_graph_dims()


    if cuda_flag:
        NN = ACNet(ndim,264,1).cuda()
    else:
        NN = ACNet(ndim,264,1)

    PATH = 'mvc_net.pt'
    NN.load_state_dict(torch.load(PATH))

    init_state,done = mvc.reset()
    pos = nx.spring_layout(init_state.g.to_networkx(), iterations=20)


    #### GCN Policy
    state = dc(init_state)
    if cuda_flag:
        state.g.ndata['x'] = state.g.ndata['x'].cuda()
    sum_r = 0
    T1 = time.time()
    [idx1,idx2] = mvc.get_ilegal_actions(state)
    it = 0


    while done == False:
        G = state.g
        [pi,val] = NN(G)
        pi = pi.squeeze()
        pi[idx1] = -float('Inf')
        pi = F.softmax(pi,dim=0)
        dist = torch.distributions.categorical.Categorical(pi)
        action = dist.sample()            
        new_state, reward, done = mvc.step(state,action)
        [idx1,idx2] = mvc.get_ilegal_actions(new_state)
        state = new_state
        sum_r += reward
        print(it)
        it += 1
    T2 = time.time()

    node_tag = state.g.ndata['x'][:,0].cpu().squeeze().numpy().tolist()

    result_GTL = -sum_r.item()
    print('GNN: {}'.format(result_GTL), file=open('output/test_result.txt','a'))

    nx.draw(state.g.to_networkx(), pos, node_color=node_tag, with_labels=True)
    plt.savefig('figures/graph_GCN.png')
    #plt.show()


    ### Greedy Policy
    state_greedy = dc(init_state)
    done = False
    sum_r2 = 0
    T1 = time.time()
    [idx1,idx2] = mvc.get_ilegal_actions(state_greedy)
    while done == False:
        G = state_greedy.g
        un_allowed = idx1.numpy().squeeze()
        degree = G.in_degrees() + G.out_degrees()
        degree[un_allowed] = 0
        degree = torch.Tensor(np.array(degree))
        action = degree.argmax()
        
        new_state, reward, done = mvc.step(state_greedy,action)
        [idx1,idx2] = mvc.get_ilegal_actions(new_state)
        state_greedy = new_state
        sum_r2 += reward
    T2 = time.time()

    result_heuristic = -sum_r2.item()
    node_tag = state_greedy.g.ndata['x'][:,0].cpu().squeeze().numpy().tolist()
    print('Heuristic: {}'.format(result_heuristic), file=open('output/test_result.txt','a'))

    nx.draw(state_greedy.g.to_networkx(), pos, node_color=node_tag, with_labels=True)
    plt.savefig('figures/graph_GCN.png')
    #plt.show()

    #print('Ratio: {}'.format((sum_r/sum_r2).item()))

    #Local Search ~~

    result_local_search = LOCALSEARCH(state_greedy)
    print('Local Search: {}'.format(min(result_local_search, result_heuristic)), file=open('output/test_result.txt','a'))


    result_GTL = GTL(state)
    print('GTL: {}'.format(min(result_GTL,result_GTL)), file=open('output/test_result.txt','a'))

    return result_GTL, result_heuristic, result_local_search, result_GTL


avg_GNN = 0
avg_Heuristic = 0
avg_LocalSearch = 0
avg_GTL = 0

iterations = 50

for i in range(iterations):
    print(f'Iteration {i+1}/{iterations}', file=open('output/test_result.txt','a'))
    a,b,c,d = f()
    avg_GNN += a
    avg_Heuristic += b
    avg_LocalSearch += min(c, b)
    avg_GTL += min(d, a)

avg_GNN /= iterations
avg_Heuristic /= iterations
avg_LocalSearch /= iterations
avg_GTL /= iterations

print('Average GNN: {}'.format(avg_GNN), file=open('output/test_result.txt','a'))
print('Average Heuristic: {}'.format(avg_Heuristic), file=open('output/test_result.txt','a'))
print('Average Local Search: {}'.format(avg_LocalSearch), file=open('output/test_result.txt','a'))
print('Average GTL: {}'.format(avg_GTL), file=open('output/test_result.txt','a'))

