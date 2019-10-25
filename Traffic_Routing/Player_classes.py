
import numpy as np
import matplotlib.pyplot as plt
import GPy


class GameData: # Data to be saved (for post processing/debugging)
     def __init__(self,N):
        self.Played_actions   = []
        self.Mixed_strategies = []  #initialize
        self.Incurred_losses   = []  #initialize
        self.Regrets          =  []  #initialize
        self.Cum_losses        = [()]*N

class Player_Hedge: # Hedge algorithm (Freund and Schapire. 1997)
    def __init__(self,K,T,min_payoff,max_payoff):
        self.type = "Hedge"
        self.K = K
        self.min_payoff = min_payoff
        self.max_payoff = max_payoff
        self.weights = np.ones(K)
        self.T  = T
        self.gamma_t = np.sqrt(8*np.log(K)/T)
        
    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)
    
    def Update(self,payoffs):
        payoffs = np.maximum(payoffs, self.min_payoff*np.ones(self.K))
        payoffs = np.minimum(payoffs, self.max_payoff*np.ones(self.K))
        payoffs_scaled = np.array((payoffs - self.min_payoff)/(self.max_payoff- self.min_payoff))
        losses = np.ones(self.K) - np.array(payoffs_scaled)        
        self.weights = np.multiply( self.weights, np.exp(np.multiply(self.gamma_t, -losses)))
        self.weights = self.weights/np.sum(self.weights) # To avoid numerical errors when the weights become too small
        

class Player_EXP3P:  # EXP3.P algorithm (Auer et al. 2002) with params according to
                     # Theorem 3.3 of [Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems, S. Bubeck, N. Cesa-Bianchi, 2012]
    def __init__(self,K,T,min_payoff,max_payoff):
        self.type = "EXP3P"
        self.K = K
        self.min_payoff = min_payoff
        self.max_payoff = max_payoff
        self.T = T
        self.weights = np.ones(K)
        self.rewards_est = np.zeros(K)

        self.beta = np.sqrt(np.log(self.K) / (self.T * self.K))
        self.gamma = 1.05*np.sqrt(np.log(self.K)*self.K/self.T)
        self.eta = 0.95*np.sqrt(np.log(self.K)/(self.T*self.K))
        assert  self.K == 1 or (self.beta > 0 and self.beta < 1 and self.gamma > 0 and self.gamma < 1)
        
    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)
    
    def Update(self, played_a, payoff):
        prob =  self.weights[played_a] / np.sum(self.weights)
        payoff = np.maximum(payoff, self.min_payoff)
        payoff = np.minimum(payoff, self.max_payoff)
        payoff_scaled = np.array((payoff - self.min_payoff)/(self.max_payoff- self.min_payoff) )

        self.rewards_est = self.rewards_est + self.beta*np.divide(np.ones(self.K), self.weights / np.sum(self.weights))
        self.rewards_est[played_a] = self.rewards_est[played_a] + payoff_scaled/prob
   
        self.weights =  np.exp(np.multiply(self.eta, self.rewards_est))
        self.weights = self.weights/np.sum(self.weights) 
        self.weights = (1- self.gamma)*self.weights  + self.gamma/self.K *np.ones(self.K)
        

class Player_GPMW:
    def __init__(self,K,T,min_payoff,max_payoff, my_strategy_vecs , kernel):
        self.type = "GPMW"
        self.K = K
        self.min_payoff = min_payoff
        self.max_payoff = max_payoff
        self.weights = np.ones(K)
        self.T  = T
        self.idx_nonzeros = np.where(np.sum(my_strategy_vecs, axis = 0) != 0)[0]

        self.mean_rewards_est = np.zeros(K)
        self.std_rewards_est = np.zeros(K)
        self.ucb_rewards_est = np.zeros(K)
        self.gamma_t = np.sqrt(8*np.log(K)/T)
        self.kernel = kernel
        self.strategy_vecs = my_strategy_vecs

        self.history_payoffs = np.empty((0, 1))
        self.history = np.empty((0, len(self.idx_nonzeros) * 2))

        self.demand = np.max(my_strategy_vecs[0])

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)


    def Update(self, played_action, total_occupancies, payoff, sigma_e , losses_hindsight):

        delta = 0.01
        t = np.size(self.history_payoffs,0) + 1
        beta_t = 2.0

        if len(self.history_payoffs) > 0:
            m = GPy.models.GPRegression(self.history, self.history_payoffs, self.kernel)
            m.Gaussian_noise.fix(sigma_e**2)

        other_occupancies = total_occupancies[self.idx_nonzeros] - self.strategy_vecs[played_action][self.idx_nonzeros]
        for a1 in range(self.K):
            x1 = self.strategy_vecs[a1][self.idx_nonzeros]
            x2 = other_occupancies + x1
            if len(self.history_payoffs) == 0:
                mu = 0
                var = self.kernel.K(np.concatenate((x1.T,x2.T), axis = 1) , np.concatenate((x1.T,x2.T), axis = 1))
            else:
                mu, var = m.predict(np.concatenate((x1.T,x2.T), axis = 1))
            sigma = np.sqrt(np.maximum(var, 1e-6))

            self.ucb_rewards_est[a1] =  mu + beta_t* sigma
            self.mean_rewards_est[a1] =  mu
            self.std_rewards_est[a1] =   sigma

        
        payoffs = np.array(self.ucb_rewards_est)
        payoffs = np.maximum(payoffs, self.min_payoff*np.ones(self.K))
        payoffs = np.minimum(payoffs, self.max_payoff*np.ones(self.K))
        payoffs_scaled = np.array((payoffs - self.min_payoff)/(self.max_payoff- self.min_payoff))
        losses = np.ones(self.K) - np.array(payoffs_scaled)        
        self.weights = np.multiply( self.weights, np.exp(np.multiply(self.gamma_t, -losses)))
        self.weights = self.weights/np.sum(self.weights) # To avoid numerical errors when the weights become too small


        self.history_payoffs = np.vstack((self.history_payoffs, payoff))
        self.history = np.vstack( (self.history , np.concatenate((self.strategy_vecs[played_action][self.idx_nonzeros].T, total_occupancies[self.idx_nonzeros].T), axis=1)))


class Player_QBRI:
    def __init__(self, K, N, T, min_payoff, max_payoff, my_strategy_vecs):
        self.type = "QBRI"
        self.K = K
        self.N = N
        self.min_payoff = min_payoff
        self.max_payoff = max_payoff
        self.weights = np.ones(K)
        self.T = T
        self.idx_nonzeros = np.where(np.sum(my_strategy_vecs, axis=0) != 0)[0]

        self.strategy_vecs = my_strategy_vecs

        self.history_payoffs = np.empty((0, 1))
        self.history_occupancies = np.empty((0, len(self.idx_nonzeros)))
        self.history_actions = np.empty((0, 1))


        self.demand = np.max(my_strategy_vecs[0])

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)

    def Q_values(self, total_occupancies ):
        values = np.zeros(self.K)
        idx1 = np.where((self.history_occupancies == total_occupancies).all(axis=1))
        for i in range(self.K):
            idx2 = np.where((self.history_actions == i).all(axis = 1))
            idx = np.intersect1d(idx1,idx2)
            count = 1
            for j in range(len(idx)):
                lambda_t = count**(-1)
                values[i] = values[i] + lambda_t*( self.history_payoffs[idx[j]] - values[i])
                count = count + 1
        return values

    def Update(self,  played_action, total_occupancies, payoff, losses_hindsight):

        self.history_payoffs = np.vstack((self.history_payoffs, payoff))
        self.history_occupancies = np.vstack((self.history_occupancies, total_occupancies[self.idx_nonzeros].T))
        self.history_actions  =  np.vstack((self.history_actions, played_action))

        zeta = 0.3
        t = np.size(self.history_payoffs, 0) + 1
        memory = 3
        epsilon_t = 1/8 * t**(-1/(self.N*memory))

        cum_reward = np.zeros(self.K)
        for i in range( np.minimum(memory, len(self.history_occupancies)) ):
            cum_reward = cum_reward + self.Q_values(self.history_occupancies[t-2-i])

        best_a =  np.argmax(cum_reward)
        self.weights = np.zeros(self.K)
        self.weights[played_action] = zeta
        self.weights[best_a] = 1- zeta
        self.weights = (1- epsilon_t)*self.weights + epsilon_t*np.ones(self.K)





        
        
        