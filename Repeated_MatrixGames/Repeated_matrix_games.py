

import numpy as np
import matplotlib.pyplot as plt
from aux_functions import Assign_payoffs, Player_MWU, Player_EXP3, Player_random, Player_GPMW
import pickle

plt.close('all')

N = 2       # number of players
K = 30      # number of actions for each player
T = 200     # time horizon,   should be at least K*log(K) to have a meaningfull EXP3.P algorithm
sigma = 1



" Data to be saved (for post processing/plotting) "
class GameData:
     def __init__(self, T,K):
        self.Played_actions = []
        self.Mixed_strategies =  []
        self.Obtained_payoffs = []
        self.Cum_payoffs =  []
        self.Regrets =  []
        self.arms = K

k1 = 1
k2 = K/5
def Kernel(a,b):
        l = k2
        return k1*np.exp( -0.5 * 1/(l**2) *np.linalg.norm( np.array(a) - np.array(b) ,2)**2)        
def Generate_A(K):
    A = []
    Cov = np.zeros([K*K,K*K])
    vector = np.zeros([K*K,2])
    idx = 0
    for a1 in range(K):
        for a2 in range(K):
            vector[idx] = [a1,a2]
            idx = idx + 1
                
    for i in range(K*K):
        for j in range(K*K):
            Cov[i,j] = Kernel(vector[i], vector[j])
                        
    Mu = 0*np.ones(K*K)
    Realiz = np.random.multivariate_normal(Mu, Cov)

    # Compute Posterior Mean (has bounded RKHS norm w.r.t. Kernel)
    post_mean = np.zeros(K*K)
    C = Cov +  sigma**2*np.eye(K*K)
    for i in range(K*K):
        B = Cov[i,:]
        post_mean[i] = B.dot(np.linalg.inv(C).dot(Realiz))
    #plt.figure()
    #plt.plot(Realiz)
    #plt.plot(post_mean)

    Matrix = np.reshape(post_mean, (K,K))
    #Matrix = Matrix - np.min(Matrix)
    #Matrix = Matrix/np.max(np.absolute(Matrix))
    A.append(Matrix)
    A.append(-Matrix)
        
    return A 

def RunGame(N,K,T,sigma, A, types , kernel_k1, kernel_k2):
    
    
    noises = np.random.normal(0,sigma,T)


    Game_data = GameData(T,K)
    Game_data.Std_matrices = []
    Game_data.Mean_matrices = []

    Player = list(range(N))  #list of all players 
    min_payoff = []
    payoffs_range = []
    for i in range(N):
        min_payoff.append( np.array(A[i].min()))
        payoffs_range.append( np.array(A[i].max() - A[i].min()) )
        Game_data.Cum_payoffs.append( np.zeros(K) )

        if types[i] == 'random':
            Player[i] = Player_random(K)
        if types[i] == 'MWU':
            Player[i] = Player_MWU(K,T,min_payoff[i],payoffs_range[i])
        if types[i] == 'EXP3':
            Player[i] = Player_EXP3(K,T,min_payoff[i],payoffs_range[i])
        if types[i] == 'GPMW':
            Player[i] = Player_GPMW(K,T,i,min_payoff[i], payoffs_range[i],  kernel_k1, kernel_k2)
    " Repated Game "
    for t in range(T): 
        " Compute played actions "
        Game_data.Played_actions.append(  [None]*N )     #initialize
        Game_data.Mixed_strategies.append(  [None]*N )   #initialize
        Game_data.Regrets.append(  [None]*N )            #initialize

        for i in range(N):
            Game_data.Mixed_strategies[t][i] = np.array(Player[i].mixed_strategy())
            Game_data.Played_actions[t][i] =   np.random.choice(range(K), p = Game_data.Mixed_strategies[t][i] )
            
        " Assign payoffs and compute regrets"
        Game_data.Obtained_payoffs.append(  [None]*N)       #initialize
        for i in range(N):
            Game_data.Obtained_payoffs[t][i] = Assign_payoffs(Game_data.Played_actions[t], A[i] )

            for a in range(K):
                modified_outcome = np.array(Game_data.Played_actions[t])
                modified_outcome[i] = a
                Game_data.Cum_payoffs[i][a] = np.array(Game_data.Cum_payoffs[i][a] + Assign_payoffs( modified_outcome, A[i] ))
            
            Game_data.Regrets[t][i] = ( np.max(Game_data.Cum_payoffs[i]) -  sum([  Game_data.Obtained_payoffs[x][i] for x in range(t+1)]) ) / (t+1)
            
        
        " Update players next mixed strategy "
        for i in range(N):
            if Player[i].type == "MWU" :
                if i == 0:                        
                    Player[i].Update(  A[i][:,Game_data.Played_actions[t][1]] )
                else:
                    Player[i].Update(  A[i][Game_data.Played_actions[t][0],:] )
                
            if Player[i].type == "EXP3" :
                noisy_payoff = Game_data.Obtained_payoffs[t][i] + np.random.normal(0, sigma, 1)
                Player[i].Update( Game_data.Played_actions[t][i], noisy_payoff, t+1 )
              

            if Player[i].type == "GPMW":
                history_actions = [Game_data.Played_actions[x][:]  for x in range(t+1) ]
                history_payoffs = [Game_data.Obtained_payoffs[x][i] + noises[x] for x in range(t+1) ]
                Player[i].GP_update( history_actions, history_payoffs , sigma)
                
                Game_data.Std_matrices.append( np.array(Player[i].std_matrix) )
                Game_data.Mean_matrices.append( np.array(Player[i].mean_matrix))
                                
                opponent_actions = [Game_data.Played_actions[x][abs(i-1)] for x in range(t+1)]
                Player[i].Update(t , opponent_actions)
        
        Game_data.A = A
    return Game_data , Player


" --------------------------------- Begin Simulations --------------------------------- "

Runs = 10


N_types = []
#N_types.append( ['GPMW', 'EXP3'])
N_types.append( ['MWU', 'random'])
N_types.append(['EXP3', 'random'])
N_types.append(['GPMW', 'random'])

avg_Regrets_P1 = []
avg_Regrets_P2 = []
std_Regrets_P1 = []
std_Regrets_P2 = []
for i in range(len(N_types)):
    np.random.seed(4)
    Regrets_P1 = [None]*Runs
    Regrets_P2 = [None]*Runs
    for run in range(Runs):
        A = Generate_A(K)  #Generate random payoff matrix

        kernel_k1 = k1
        kernel_k2 = k2
            
        
        Games_data, Player  = RunGame(N,K,T,sigma, A, N_types[i] , kernel_k1, kernel_k2)
        Regrets_P1[run] = np.array([  Games_data.Regrets[x][0] for x in range(T)])
        Regrets_P2[run] = np.array([  Games_data.Regrets[x][1] for x in range(T)])
        print('Run: ' + str(run))
    avg_Regrets_P1.append( np.mean(Regrets_P1,0) )
    avg_Regrets_P2.append( np.mean(Regrets_P2,0) )
    std_Regrets_P1.append( np.std(Regrets_P1,0) )
    std_Regrets_P2.append( np.std(Regrets_P2,0) )


" SAVING "
if N_types[0] == ['GPMW', 'EXP3']:
    with open('stored_GPMW_vs_Exp3.pckl', 'wb') as file:
        pickle.dump(N_types , file)
        pickle.dump(avg_Regrets_P1, file)
        pickle.dump(std_Regrets_P1, file)
        pickle.dump(avg_Regrets_P2, file)
        pickle.dump(std_Regrets_P2, file)
else:
    with open('stored_against_random.pckl', 'wb') as file:
        pickle.dump(N_types , file)
        pickle.dump(avg_Regrets_P1, file)
        pickle.dump(std_Regrets_P1, file)
        pickle.dump(avg_Regrets_P2, file)
        pickle.dump(std_Regrets_P2, file)
