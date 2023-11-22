import sys
import os
from discrete_actor_critic import DiscreteActorCritic
from MVC import MVC
import matplotlib.pyplot as plt
from smooth_signal import smooth
import numpy as np
import time
import torch

n = 30  # number of nodes
p = 0.15 # edge probability
env = MVC(n,p)
cuda_flag = False
alg = DiscreteActorCritic(env,cuda_flag)

num_episodes = int(input("Enter number of episodes: "))

print('hey', file=open('output/output.txt','w'))
print(f'env.g: {env.init_state.g}', file=open('output/output.txt','a'))

for i in range(num_episodes):
    #print('Epoch: {}'.format(i))
    T1 = time.time()
    log = alg.train()
    T2 = time.time()
    print('Epoch: {}. total_return: {}. TD error: {}. H: {}. T: {}'.format(i,np.round(log.get_current('tot_return'),2),np.round(log.get_current('TD_error'),3),np.round(log.get_current('entropy'),3),np.round(T2-T1,3)),file=open('output/output2.txt','a'))

Y = np.asarray(log.get_log('tot_return'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))  
fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(x, Y , Y2)
plt.xlabel('episodes')
plt.ylabel('episode return')

Y = np.asarray(log.get_log('TD_error'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig3 = plt.figure()
ax3 = plt.axes()
ax3.plot(x, Y , Y2)
plt.xlabel('episodes')
plt.ylabel('mean TD error')

Y = np.asarray(log.get_log('entropy'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig4 = plt.figure()
ax4 = plt.axes()
ax4.plot(x, Y , Y2)
plt.xlabel('episodes')
plt.ylabel('mean entropy')

if not os.path.exists('figures'):
    os.makedirs('figures')

fig2.savefig('figures/episode_return.png')
fig3.savefig('figures/mean_TD_error.png')
fig4.savefig('figures/mean_entropy.png')

#plt.show()

PATH = 'mvc_net.pt'
torch.save(alg.model.state_dict(),PATH)
