import matplotlib.pyplot as plt
import pickle
import numpy as np

T = 200
markers = ['*', 'o', 'x', '^']


############### AGAINST RANDOM OPPONENT #############
with open('stored_against_random.pckl', 'rb') as file:
    N_types = pickle.load(file)
    avg_Regrets_P1 = pickle.load(file)
    std_Regrets_P1 = pickle.load(file)
    avg_Regrets_P2 = pickle.load(file)
    std_Regrets_P2 = pickle.load(file)

fig = plt.figure(figsize=(7, 4.5))
plt.title('Average Regret')
idx_marker = 0
for i in range(len(N_types)):
    if N_types[i][0] == 'MWU':
        p = plt.plot(np.arange(T), avg_Regrets_P1[i], linestyle=':')
    else:
        p = plt.plot(np.arange(T), avg_Regrets_P1[i], marker = markers[idx_marker], markevery = 10, markersize = 7)
    color = p[0].get_color()
    plt.fill_between(range(T), avg_Regrets_P1[i] - std_Regrets_P1[i], avg_Regrets_P1[i] + std_Regrets_P1[i], alpha=0.1,
                     color=color)
    idx_marker += 1

plt.legend(['Hedge', 'Exp3.P', 'GP-MW'], prop={'size': 18})
plt.xlabel('time')
plt.ylim(0, 1.6)
fig.tight_layout()
plt.rcParams.update({'font.size': 14})


############### GP-HEDGE VS EXP3.P #############
with open('stored_GPMW_vs_Exp3.pckl', 'rb') as file:
    N_types = pickle.load(file)
    avg_Regrets_P1 = pickle.load(file)
    std_Regrets_P1 = pickle.load(file)
    avg_Regrets_P2 = pickle.load(file)
    std_Regrets_P2 = pickle.load(file)

fig = plt.figure(figsize=(7,4.5))
plt.title('Average Regret')
for i in range(1):
    p = plt.plot(np.arange(T), avg_Regrets_P1[i], color='C2', marker = markers[2], markevery = 10 , markersize = 7)
    color = p[0].get_color()
    plt.fill_between(range(T), avg_Regrets_P1[i] - std_Regrets_P1[i], avg_Regrets_P1[i] + std_Regrets_P1[i], alpha=0.1,
                         color=color)
    p = plt.plot(np.arange(T), avg_Regrets_P2[i], color='C1', marker = markers[1], markevery = 10 , markersize = 7)
    color = p[0].get_color()
    plt.fill_between(range(T), avg_Regrets_P2[i] - std_Regrets_P2[i], avg_Regrets_P2[i] + std_Regrets_P2[i], alpha=0.1,
                         color=color)
plt.legend(['GP-MW', 'Exp3.P'],  prop={'size': 18})
plt.xlabel('time')
plt.ylim(0, 1.5)
fig.tight_layout()
plt.rcParams.update({'font.size': 14})


