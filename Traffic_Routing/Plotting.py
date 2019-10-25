import pickle
import matplotlib.pyplot as plt
import numpy as np

outer_Runs = 10
num_controlled_players_range = [5,10,20,50,100,200]

Algos = ['Hedge', 'EXP3P' , 'GPMW_degree4', 'GPMW_degree6', 'GPMW_degree2' ,'QBRI']
Legend = ['Hedge', 'Exp3.P', 'GP-MW (dgr 4)', 'GP-MW (dgr 6)' ,'GP-MW (dgr 2)', 'Q-BRI']
markers = ['*', 'x', 'o', '^', 'd', 's']
figs_size = (7,4.5)
#####   Congestions as function of # of learning agents   ######
fig = plt.figure(figsize=figs_size)
plt.rcParams.update({'font.size': 14})
idx_marker = 0
for Algo in Algos:
    avg_mean_congestion = []
    for p in num_controlled_players_range:
        mean_congestion = []
        for outrun in range(outer_Runs):
            file = open('Saved_computations/store_' + Algo + '_controlled_' + str(p) + '_run_' + str(outrun) + '.pckl', 'rb')
            loader = pickle.load(file)
            file.close()
            mean_congestion.append(np.mean(loader[3][-5:]))
        avg_mean_congestion.append(np.mean(mean_congestion))
    if Algo == 'Hedge':
        plt.plot(num_controlled_players_range, avg_mean_congestion, linestyle=':', marker='.')
    else:
        plt.plot(num_controlled_players_range, avg_mean_congestion, linestyle='-', marker= markers[idx_marker], linewidth = 2, markersize = 10)
        idx_marker += 1
plt.title('Average final Congestion')
plt.xlabel('# of learning agents')
fig.tight_layout()
plt.savefig('Fig4.png', dpi = 100)

#####   Final Regrets as function of # of learning agents   ######
fig = plt.figure(figsize=figs_size)
idx_marker = 0
for Algo in Algos:
    avg_mean_congestion = []
    for p in num_controlled_players_range:
        mean_congestion = []
        for outrun in range(outer_Runs):
            file = open('Saved_computations/store_' + Algo + '_controlled_' + str(p) + '_run_' + str(outrun) + '.pckl', 'rb')
            loader = pickle.load(file)
            file.close()
            mean_congestion.append(np.mean(loader[1][-5:]))
        avg_mean_congestion.append(np.mean(mean_congestion))
    if Algo == 'Hedge':
        plt.plot(num_controlled_players_range, avg_mean_congestion,  linestyle=':', marker='.')
    else:
        plt.plot(num_controlled_players_range, avg_mean_congestion,  linestyle='-', marker= markers[idx_marker], linewidth = 2, markersize = 10)
        idx_marker += 1
plt.title('Average final Regret')
plt.xlabel('# of learning agents')
fig.tight_layout()
plt.rcParams.update({'font.size': 14})
plt.savefig('Fig3.png', dpi = 100)

#####   Time-averaged Regrets as function of time, for 100 learning agents   ######
fig = plt.figure(figsize=figs_size)
idx_marker = 0
for Algo in Algos:
    avg_mean_congestion = []
    for p in [num_controlled_players_range[4]]:
        mean_congestion = []
        for outrun in range(outer_Runs):
            file = open('Saved_computations/store_' + Algo + '_controlled_' + str(p) + '_run_' + str(outrun) + '.pckl', 'rb')
            loader = pickle.load(file)
            file.close()
            mean_congestion.append(np.array(loader[1]))
        avg_mean_congestion.append(np.mean(np.array(mean_congestion), axis = 0))
    if Algo == 'Hedge':
        plt.plot(avg_mean_congestion[0], linestyle=':')
    else:
        plt.plot(avg_mean_congestion[0], linewidth = 2, marker= markers[idx_marker], markevery = 10, markersize = 10)
        idx_marker += 1
plt.title('Average Regret')
plt.xlabel('time')
plt.rcParams["legend.loc"] = 'upper right'
plt.legend(Legend,  prop={'size': 15}, loc = (0.001,0.001))
fig.tight_layout()
plt.rcParams.update({'font.size': 14})
plt.savefig('Fig1.png', dpi = 100)

#####   Congestions as function of time , for 100 learning agents  ######
fig = plt.figure(figsize=figs_size)
idx_marker = 0
for Algo in Algos:
    avg_mean_congestion = []
    for p in [num_controlled_players_range[4]]:
        mean_congestion = []
        for outrun in range(outer_Runs):
            file = open('Saved_computations/store_' + Algo + '_controlled_' + str(p) + '_run_' + str(outrun) + '.pckl', 'rb')
            loader = pickle.load(file)
            file.close()
            mean_congestion.append(loader[3])
        avg_mean_congestion.append(np.mean(np.array(mean_congestion), axis = 0))
    if Algo == 'Hedge':
        plt.plot(avg_mean_congestion[0], linestyle=':')
    else:
        plt.plot(avg_mean_congestion[0], linewidth = 2, marker= markers[idx_marker], markevery = 10, markersize = 10)
        idx_marker += 1
plt.title('Average Congestion')
plt.xlabel('time')
fig.tight_layout()
plt.rcParams.update({'font.size': 14})
plt.savefig('Fig2.png', dpi = 100)

