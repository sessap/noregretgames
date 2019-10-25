
import numpy as np

" Type of Players "

class Player_random:
    def __init__(self,K):
        self.type = "random"
        self.K = K
        self.weights = np.ones(K)
        
    def mixed_strategy(self):        
        return self.weights / np.sum(self.weights)
    
    def Update(self,payoffs):
        self.weights = self.weights 
        
class Player_MWU:  # Hedge algorithm (Freund and Schapire. 1997)
    def __init__(self,K,T,min_payoff,payoffs_range):
        self.type = "MWU"
        self.K = K
        self.min_payoff = min_payoff
        self.payoffs_range = payoffs_range
        self.weights = np.ones(K)
        self.T  = T
        self.gamma_t = np.sqrt(8*np.log(K)/T)
        
    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)
    
    def Update(self,payoffs):
        payoffs = np.maximum(payoffs, self.min_payoff*np.ones(self.K))
        payoffs = np.minimum(payoffs, (self.min_payoff + self.payoffs_range)*np.ones(self.K))
        payoffs_scaled = np.array((payoffs - self.min_payoff)/self.payoffs_range)
        losses = np.ones(self.K) - np.array(payoffs_scaled)        
        self.weights = np.multiply( self.weights, np.exp(np.multiply(self.gamma_t, -losses)))
        self.weights = self.weights/np.sum(self.weights) # To avoid numerical errors when the weights become too small

class Player_EXP3:   # EXP3.P algorithm (Auer et al. 2002) with params according to
                     # Theorem 3.3 of [Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems, S. Bubeck, N. Cesa-Bianchi, 2012]
    def __init__(self,K,T,min_payoff,payoffs_range):
        self.type = "EXP3"
        self.K = K
        self.min_payoff = min_payoff
        self.payoffs_range = payoffs_range
        self.T = T
        self.weights = np.ones(K)
        self.rewards_est = np.zeros(K)

        delta = 0.01
        self.beta = np.sqrt(np.log(self.K * (1/delta) )/(self.T*self.K))
        self.gamma = 1.05*np.sqrt(np.log(self.K)*self.K/self.T)
        self.eta = 0.95*np.sqrt(np.log(self.K)/(self.T*self.K))
        assert  self.beta > 0 and self.beta < 1 and self.gamma > 0 and self.gamma < 1
        
    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)
    
    
    def Update(self, played_a, payoff, t):
        prob =  self.weights[played_a] / np.sum(self.weights)
        payoff = np.maximum(payoff, self.min_payoff)
        payoff = np.minimum(payoff, (self.min_payoff + self.payoffs_range))
        payoff_scaled = np.array((payoff - self.min_payoff)/self.payoffs_range)
        
        self.rewards_est = self.rewards_est + self.beta*np.divide(np.ones(self.K), self.weights / np.sum(self.weights))
        self.rewards_est[played_a] = self.rewards_est[played_a] + payoff_scaled/prob
   
        self.weights =  np.exp(np.multiply(self.eta, self.rewards_est))
        self.weights = self.weights/np.sum(self.weights) 
        self.weights = (1- self.gamma)*self.weights  + self.gamma/self.K *np.ones(self.K)
         


class Player_GPMW:
    def __init__(self, K,T,i, min_payoff, payoffs_range, kernel_k1, kernel_k2):
        self.type = "GPMW"
        self.idx_player = i
        self.K = K
        self.min_payoff = min_payoff
        self.payoffs_range = payoffs_range
        self.weights = np.ones(K)
        self.UCB_matrix = np.zeros((K,K))
        self.mean_matrix = 0*np.ones((K,K))
        self.gamma_t = np.sqrt(8*np.log(K)/T)
        self.k1 = kernel_k1
        self.k2 = kernel_k2
        self.kernel_matrix = np.zeros((K*K,K*K))
        self.vec_idx = np.zeros([K*K,2])
        idx = 0
        for a1 in range(K):
            for a2 in range(K):
                self.vec_idx[idx,:] = [a1,a2] 
                idx = idx + 1
                
        for i in range(K*K):
            for j in range(K*K):
                self.kernel_matrix[i,j] = np.array(self.Kernel(self.vec_idx[i,:], self.vec_idx[j,:]))
        
        self.var_matrix = np.zeros((K,K))
        self.std_matrix = np.zeros((K,K))
        for a1 in range(K):
            for a2 in range(K):
                idx = np.squeeze(np.where(np.all(self.vec_idx == [a1,a2], axis=1)))
                self.var_matrix[a1,a2] = np.array(self.kernel_matrix[idx,idx])
        
        self.std_matrix = np.sqrt(self.var_matrix)

        
    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)
    
    def Kernel(self, a,b):
        vec1 = a
        vec2 = b

        l = self.k2
        return self.k1 * np.exp( -0.5 * 1/(l**2) *np.linalg.norm( np.array(vec1) - np.array(vec2) ,2)**2)

    
    def GP_update(self, history_actions, history_payoffs, sigma_e ):
        delta = 0.01
        t = np.size(history_actions,0)
        beta_t = 2.0

        if 0: # Non-Recursive update
            temp_UCB_matrix = np.zeros((self.K,self.K))
            temp_mean_matrix = np.zeros((self.K,self.K))
            temp_std_matrix = np.zeros((self.K,self.K))

            C = [[self.Kernel(x1,x2) for x1 in history_actions[:] ] for x2 in history_actions[:]] + sigma_e**2*np.eye(t)
            for a1 in range(self.K):
                for a2 in range(self.K):
                    A = np.squeeze(self.Kernel([a1,a2],[a1,a2]))
                    B = np.squeeze([ self.Kernel(x, [a1,a2]) for  x in history_actions[:][:] ])
                    if np.size(C) == 1 :
                        mu = C**-1 * B * history_payoffs
                    else:
                        mu = B.dot(np.linalg.inv(C).dot(history_payoffs))   #np.linalg.inv(C).dot(B.T).T.dot(history_payoffs)

                    sigma = np.sqrt( A - B.dot(np.linalg.inv(C).dot(B.T)) )
                    self.UCB_matrix[a1,a2] =  mu + beta_t* sigma
                    self.mean_matrix[a1,a2] =  mu 
                    self.std_matrix[a1,a2] =  sigma 
#                    temp_UCB_matrix[a1,a2] =  mu + beta_t* sigma
#                    temp_mean_matrix[a1,a2] =  mu 
#                    temp_std_matrix[a1,a2] =  sigma 
        else:
            mean_matrix_prev = np.array(self.mean_matrix)
            var_matrix_prev = np.array(self.var_matrix)
            kernel_matrix_prev = np.array(self.kernel_matrix)
            
            mean_matrix = np.zeros((self.K,self.K))
            var_matrix = np.zeros((self.K,self.K))

            idx2 = np.squeeze(np.where(np.all(self.vec_idx == history_actions[t-1][:], axis=1)))
            for a1 in range(self.K):
                for a2 in range(self.K):
                    idx1 = np.squeeze(np.where(np.all(self.vec_idx == [a1,a2], axis=1)))
                
                    mean_matrix[a1,a2] =  mean_matrix_prev[a1,a2] + ( kernel_matrix_prev[idx1,idx2]/(sigma_e**2 + var_matrix_prev[history_actions[t-1][0], history_actions[t-1][1]]) ) *(history_payoffs[t-1] - mean_matrix_prev[history_actions[t-1][0], history_actions[t-1][1]])
                    var_matrix[a1,a2] =  var_matrix_prev[a1,a2] - (kernel_matrix_prev[idx1,idx2]**2)/ (sigma_e**2 + var_matrix_prev[history_actions[t-1][0], history_actions[t-1][1]])
                    if var_matrix[a1,a2] < 0:
                        print(var_matrix[a1,a2])    
                    self.UCB_matrix[a1,a2] =  mean_matrix[a1,a2] + beta_t*np.sqrt(var_matrix[a1,a2])
    
            kernel_matrix = kernel_matrix_prev - np.outer(kernel_matrix_prev[:,idx2], kernel_matrix_prev[idx2,:]) * (1/ (sigma_e**2 + var_matrix_prev[history_actions[t-1][0], history_actions[t-1][1]]))

            self.var_matrix = var_matrix
            self.std_matrix = np.sqrt(var_matrix)
            self.mean_matrix = mean_matrix
            self.kernel_matrix = kernel_matrix
            
    def Update(self, t , opponents_actions):
        if self.idx_player == 0:
            payoffs = self.UCB_matrix[:,opponents_actions[t]] 
        else:
            payoffs = self.UCB_matrix[opponents_actions[t],:]
        
        payoffs = np.maximum(payoffs, self.min_payoff*np.ones(self.K))
        payoffs = np.minimum(payoffs, (self.min_payoff + self.payoffs_range)*np.ones(self.K))
        payoffs_scaled = np.array((payoffs - self.min_payoff)/self.payoffs_range)
        losses = np.ones(self.K) - np.array(payoffs_scaled)
        self.weights = self.weights * np.exp(np.multiply(self.gamma_t, - losses))            
                    
        
            
def Assign_payoffs(outcome , payoff_matrix):
    A = payoff_matrix
    return A[outcome[0],outcome[1]]
