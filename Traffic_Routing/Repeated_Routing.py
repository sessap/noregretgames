import matplotlib.pyplot as plt
plt.close()

from Network_functions import* # Load the needed Sioux Falls Network parameters and import needed functi

# IMPORT MODULES AND CLASSES
import pickle
import os
import numpy as np
import pandas
import GPy
import random
import networkx as nx
from Player_classes import GameData, Player_Hedge, Player_EXP3P, Player_GPMW, Player_QBRI
####################################################################################

SiouxNetwork, SiouxNetwork_data = Create_Network()


OD_demands = pandas.read_csv("SiouxFalls_OD_matrix.txt", header=None)
OD_demands = OD_demands.values

OD_pairs    = []
Demands         = []
freeflowtrips = []
for i in range(24):
    for j in range(24):
        if OD_demands[i,j] > 0:
            OD_pairs.append([i+1, j+1 ])
            Demands.append(OD_demands[i,j]/100)
            freeflowtrips.append(   nx.shortest_path_length(SiouxNetwork, str(i+1), str(j+1), weight = 'weight') )
            
np.random.seed(3)
random.seed(3)

Strategy_vectors = Compute_Strategy_vectors(OD_pairs, Demands, SiouxNetwork_data.Freeflowtimes, SiouxNetwork, SiouxNetwork_data.Edges)

" Parameters "
T = 100                         #number of iterations
N = len(OD_pairs)               #number of players
sigmas = np.zeros(N)            #standard deviation of noisy feedback (initialize)

Algo = 'GPMW'                  # Choose between GPMW (GP-MW in the paper), Hedge, EXP3P, or QBRI
poly_degree = 6


num_controlled_players_range = [5,10,20,50,100,200]
outer_Runs = 10
for outrun in range(outer_Runs):

    Kernels = [None] * N
    idxs_controlled =  random.sample( range(0, N),  num_controlled_players_range[0])
    for num_controlled_players in num_controlled_players_range:
        if len(idxs_controlled) < num_controlled_players:
            idxs_controlled = np.hstack( (idxs_controlled, random.sample(range(0,N),num_controlled_players-len(idxs_controlled)))   )

        " Estimate maximum_traveltimes"
        max_traveltimes = np.zeros(N)
        Outcomes = []
        Payoffs = []
        for rand_outcomes in range(10000):
            outcome = np.zeros(N) # all play first action by default
            for p in idxs_controlled:
                outcome[p] = np.random.choice(len(Strategy_vectors[p]))
            traveltimes = Compute_traveltimes(SiouxNetwork_data, Strategy_vectors, outcome.astype(int) , 'all')
            max_traveltimes = np.maximum(max_traveltimes, traveltimes )

            Outcomes.append(outcome)
            Payoffs.append(-traveltimes)

        sigmas = 0.001*max_traveltimes

        if Algo == 'GPMW':
            for p in idxs_controlled:
                " Estimate Kernels hyperparameters"
                if Kernels[p] == None: # Estimate them
                    idx_nonzeros = np.where(np.sum(Strategy_vectors[p], axis=0) != 0)[0]
                    dim = len(idx_nonzeros)


                    kernel_1 = GPy.kern.Poly(input_dim=dim, variance=sigmas[p] ** 2, scale=1., bias=1., order=1., active_dims=np.arange(0, dim))
                    kernel_2 = GPy.kern.Poly(input_dim=dim, variance=sigmas[p] ** 2, scale=1., bias=1., order=poly_degree, active_dims=np.arange(dim, 2 * dim))
                    Kernels[p] = kernel_1 * kernel_2

                    if 1:#len(kernel_params_loaded[p] ) > 1:
                        X = np.empty((0,dim*2))
                        y = np.empty((0,1))
                        for a in range(len(Outcomes)):
                            x1 = Strategy_vectors[p][int(Outcomes[a][p])]
                            x2 = np.sum([Strategy_vectors[i][int(Outcomes[a][i])] for i in range(N)], axis=0)

                            X = np.vstack( (X,  np.concatenate((x1[idx_nonzeros].T, x2[idx_nonzeros].T), axis=1)) )
                            y = np.vstack( (y, Payoffs[a][p] + np.random.normal(0, sigmas[p], 1)))
                        # Fit to data using Maximum Likelihood Estimation of the parameters
                        m = GPy.models.GPRegression(X[0:199,:], y[0:199], Kernels[p])
                        m.Gaussian_noise.fix(sigmas[p]**2)
                        m.constrain_bounded(1e-6, 1e6)
                        m.optimize_restarts(num_restarts=1, max_f_eval = 500)

                        #os.remove('store_kernel_params.pckl')
                        #with open('store_kernel_params.pckl', 'wb') as f:
                        #    pickle.dump((kernel_params), f)

        #################################################  START SIMULATION #######################################################

        Runs = 10
        Regrets = []
        additional_congestions = []
        for run in range(Runs):

            " Initialize Players "
            Players = []
            for i in range(N):
                K_i = len(Strategy_vectors[i])
                min_payoff = - max_traveltimes[i]
                max_payoff = 0
                if i in idxs_controlled and K_i > 1:
                    if Algo == 'Hedge':
                        Players.append(Player_Hedge(K_i, T, min_payoff, max_payoff))
                    elif Algo == 'EXP3P':
                        Players.append(Player_EXP3P(K_i, T, min_payoff, max_payoff))
                    elif Algo == 'GPMW':
                        Players.append( Player_GPMW(K_i, T,min_payoff,max_payoff, Strategy_vectors[i] , Kernels[i]) )
                    elif Algo == 'QBRI':
                        Players.append( Player_QBRI(K_i, N, T,min_payoff,max_payoff, Strategy_vectors[i]  ) )
                    Players[i].OD_pair = OD_pairs[i]
                else:
                    K_i = 1
                    Players.append(Player_EXP3P(K_i, T, min_payoff, max_payoff))
                    Players[i].OD_pair = OD_pairs[i]


            Game_data = GameData(N)
            for i in range(N):
                Game_data.Cum_losses[i] = np.zeros(Players[i].K)

            Total_occupancies =  []
            addit_Congestions = []

            for t in range(T):

                " Compute played actions "
                mixed_strategy_t = []
                played_actions_t = []
                for i in range(N):
                    mixed_strategy_t.append( np.array(Players[i].mixed_strategy()) )
                    played_actions_t.append( np.random.choice(range(Players[i].K), p = np.array(mixed_strategy_t[i]) ) )
                Game_data.Mixed_strategies.append(mixed_strategy_t)
                Game_data.Played_actions.append(played_actions_t)

                " Assign payoffs and compute regrets"
                losses_t = Compute_traveltimes(SiouxNetwork_data, Strategy_vectors, Game_data.Played_actions[t], 'all')
                Game_data.Incurred_losses.append(  losses_t )


                if 1:  # compute regrets
                    regrets_t = []
                    losses_hindsight_t = []
                    for i in range(N):
                        if Players[i].K > 1:
                            losses_hindsight_t.append(np.zeros(Players[i].K))
                            for a in range(Players[i].K):
                                modified_outcome = np.array(Game_data.Played_actions[t])
                                modified_outcome[i] = a
                                losses_hindsight_t[i][a] = Compute_traveltimes(SiouxNetwork_data, Strategy_vectors, modified_outcome , i)

                            Game_data.Cum_losses[i] +=  np.array(losses_hindsight_t[i])
                            regrets_t.append(  ( sum([  Game_data.Incurred_losses[x][i] for x in range(t+1)]) - np.min(Game_data.Cum_losses[i]) ) / (t+1) )
                        else:
                            losses_hindsight_t.append(Game_data.Incurred_losses[t][i])
                            regrets_t.append(0)
                    Game_data.Regrets.append(regrets_t)
                else:
                    Game_data.Regrets.append(np.zeros(N))


                Total_occupancies.append( np.sum([Strategy_vectors[i][Game_data.Played_actions[t][i]] for i in range(N)], axis = 0) )

                E = np.size(SiouxNetwork_data.Edges,0)
                a = SiouxNetwork_data.Freeflowtimes
                b = np.divide( np.multiply(a , 0.15*np.ones((E,1))) , np.power(SiouxNetwork_data.Capacities, SiouxNetwork_data.Powers))
                addit_Congestions.append(  np.divide( np.multiply(b, np.power( Total_occupancies[t], SiouxNetwork_data.Powers) ), a) )


                " Update players mixed strategies "
                for i in range(N):
                    if Players[i].type == "EXP3P" :
                            noisy_loss = Game_data.Incurred_losses[t][i] + np.random.normal(0, sigmas[i], 1)
                            Players[i].Update( Game_data.Played_actions[t][i], -noisy_loss )

                    if Players[i].type == "Hedge" :
                            noisy_losses = losses_hindsight_t[i]
                            Players[i].Update( -noisy_losses )

                    if Players[i].type == "GPMW" :
                            noisy_loss = Game_data.Incurred_losses[t][i] + np.random.normal(0, sigmas[i], 1)
                            Players[i].Update( Game_data.Played_actions[t][i], Total_occupancies[-1], -noisy_loss, sigmas[i] , losses_hindsight_t[i])

                    if Players[i].type == "QBRI":
                        noisy_loss = Game_data.Incurred_losses[t][i] + np.random.normal(0, sigmas[i], 1)
                        Players[i].Update(Game_data.Played_actions[t][i], Total_occupancies[-1], -noisy_loss ,losses_hindsight_t[i])

            ############## Save data over multiple runs ##################
            Regrets.append(Game_data.Regrets)
            additional_congestions.append(np.squeeze(addit_Congestions))

            if 1:
                del(Game_data)
                del(Players)

        mean_regrets = np.mean(Regrets, axis = 0)
        avg_mean_regrets = np.mean(mean_regrets[:,idxs_controlled], axis = 1)
        mean_additional_congestions = np.mean(additional_congestions, axis = 0)
        avg_mean_additional_congestions = np.mean(mean_additional_congestions, axis = 1)


        if 0:
            if Algo == 'GPMW':
                file = open('Saved_computations/store_'+ Algo +'_degree' + str(int(poly_degree))+ '_controlled_'+ str(num_controlled_players) + '_run_'+ str(outrun) + '.pckl', 'wb')
            else:
                file = open('Saved_computations/store_'+ Algo + '_controlled_'+ str(num_controlled_players) + '_run_'+ str(outrun) + '.pckl', 'wb')
            pickle.dump((mean_regrets[:,idxs_controlled] ,avg_mean_regrets, mean_additional_congestions,avg_mean_additional_congestions), file)
            file.close()
    print('Concluded Outer Run:' + str(outrun))