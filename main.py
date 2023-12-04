import sys
import os
from discrete_actor_critic import DiscreteActorCritic
from MVC import MVC
import matplotlib.pyplot as plt
from smooth_signal import smooth
import numpy as np
import time
import torch


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(torch.cuda.is_available())
cuda_flag = False   # CUDA

n = 250  # number of nodes
p = 0.03 # edge probability
env = MVC(n,p)
alg = DiscreteActorCritic(env,cuda_flag)

num_episodes = 100

for i in range(num_episodes):
    T1 = time.time()
    log = alg.train()
    T2 = time.time()
    print('Epoch: {}. total_return: {}. TD error: {}. H: {}. T: {}'.format(i,np.round(log.get_current('tot_return'),2),np.round(log.get_current('TD_error'),3),np.round(log.get_current('entropy'),3),np.round(T2-T1,3)),file=open('output/output2.txt','a'))
    print(f'time it took: {T2-T1}')
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

PATH = 'models\mvc_net_' + str(n) + '_' + str(p) + '_' + str(num_episodes) + '.pt'
torch.save(alg.model.state_dict(),PATH)