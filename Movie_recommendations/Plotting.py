import matplotlib.pyplot as plt
import pickle
import numpy as np

markersize = 10

adaptive_adv = 1
with open('saved_Regrets_adaptive_adv'+ str(adaptive_adv)+'.pckl', 'rb') as f:
    Regrets_GPMW = pickle.load(f)
    Regrets_GPUCB = pickle.load(f)
    Regrets_StableOpt = pickle.load(f)

fig = plt.figure(figsize=(7,4.5))
plt.title('Average Regret')
plt.xlabel('iterations')
plt.plot(np.mean(Regrets_GPMW , axis = 0),  color = 'green' , marker = '.' , markevery = 30, markersize = markersize)
plt.plot(np.mean(Regrets_GPUCB , axis = 0), color = 'black' , marker = '*' , markevery = 30, markersize = markersize)
plt.plot(np.mean(Regrets_StableOpt, axis = 0), color = 'red', marker = 'x', markevery = 30, markersize = markersize)
plt.legend(['GP-MW', 'GP-UCB', 'StableOpt'], prop={'size': 20})
fig.tight_layout()
plt.rcParams.update({'font.size': 14})
plt.savefig('Against_adaptive.png')

adaptive_adv = 0
with open('saved_Regrets_adaptive_adv'+ str(adaptive_adv)+'.pckl', 'rb') as f:
    Regrets_GPMW = pickle.load(f)
    Regrets_GPUCB = pickle.load(f)
    Regrets_StableOpt = pickle.load(f)

fig = plt.figure(figsize=(7,4.5))
plt.title('Average Regret')
plt.xlabel('iterations')
plt.plot(np.mean(Regrets_GPMW , axis = 0),  color = 'green' , marker = '.' , markevery = 30, markersize = markersize)
plt.plot(np.mean(Regrets_GPUCB , axis = 0), color = 'black' , marker = '*' , markevery = 30, markersize = markersize)
plt.plot(np.mean(Regrets_StableOpt, axis = 0), color = 'red', marker = 'x', markevery = 30, markersize = markersize)
plt.legend(['GP-MW', 'GP-UCB', 'StableOpt'], prop={'size': 20})
fig.tight_layout()
plt.rcParams.update({'font.size': 14})
plt.savefig('Against_random.png')
