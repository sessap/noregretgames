
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
import random
####################################################################

with open('rating_features_and_matrix.pckl', 'rb') as f:
    rating_Matrix = pickle.load(f)
    users_features = pickle.load(f)
    movies_features = pickle.load(f).T

N_movies = np.size(movies_features, 0)
N_users = np.size(users_features, 0)

def f(user, movie):
    return rating_Matrix[int(user), int(movie)]


N_groups = 80 #number of groups
kmeans = KMeans(n_clusters= N_groups, random_state=0).fit(movies_features)
group_labels = kmeans.labels_
Groups_idx = []
for group in range(N_groups):
    idx_movies = np.where( group_labels == group)[0]
    Groups_idx.append(idx_movies)


random.seed(3)
np.random.seed(3)

Users = np.array( random.sample( range(0, N_users),  N_users) )


""" EXPERIMENT PARAMETERS """
Runs  = 10
T = 1000
for adaptive_adv in [1]:
    """"""""""""""""""""""""""""""""""""

    Regrets_GPMW = []
    Regrets_GPUCB = []
    Regrets_StableOpt = []
    for run in range(Runs):
        X = []
        y = []
        for idx_user in range(len(Users)):
            X.append([])
            y.append([])
            for i in range(10):
                chosen_movie = int( random.sample( range(0, N_movies),  1)[0] )
                chosen_user = Users[idx_user]
                idx = int( np.where( Users == chosen_user )[0] )
                X[idx_user].append(  movies_features[chosen_movie, :])
                y[idx_user].append(   f(chosen_user, chosen_movie ) )

        kernel = 1*DotProduct(sigma_0=0)
        GPs = []
        for i in range(len(Users)):
            GPs.append(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 0 ) )
            if 1:
                GPs[i].fit(X[i], y[i])


        UCB_gpmw_matrix = np.empty(( 0, N_movies))
        LCB_stableopt_matrix = np.empty(( 0, N_movies))
        for i in range(len(GPs)):
            payoffs = np.zeros(N_movies)
            lcb_payoffs = np.zeros(N_movies)
            for movie in range(N_movies):
                mu, sigma = GPs[i].predict(movies_features[movie, :].reshape(1, -1), return_std=True)
                beta_t = 2.0
                payoffs[movie] = mu + beta_t * sigma
                lcb_payoffs[movie] = mu - beta_t * sigma
            UCB_gpmw_matrix = np.vstack((UCB_gpmw_matrix, payoffs))
            LCB_stableopt_matrix = np.vstack((LCB_stableopt_matrix, lcb_payoffs))

        UCB_gpucb_matrix = np.array(UCB_gpmw_matrix)
        UCB_stableopt_matrix = np.array(UCB_gpmw_matrix)

        X_gpmw = X.copy()
        y_gpmw = y.copy()
        X_ucb = X.copy()
        y_ucb = y.copy()
        X_stableopt = X.copy()
        y_stableopt = y.copy()

        #################################


        mixed_strategy_GPMW = np.ones(N_movies)/N_movies
        GPMW_payoffs = []
        GPMW_cum_f = np.zeros(N_movies)
        GPMW_regrets = []

        GPUCB_payoffs = []
        GPUCB_cum_f = np.zeros(N_movies)
        GPUCB_regrets = []

        StableOpt_payoffs = []
        StableOpt_cum_f = np.zeros(N_movies)
        StableOpt_regrets = []

        mixed_strategy_adversary_gpmw = np.ones(len(Users))/len(Users)
        mixed_strategy_adversary_ucb = np.ones(len(Users))/len(Users)
        mixed_strategy_adversary_stableopt = np.ones(len(Users))/len(Users)

        for t in range(T):

            if adaptive_adv:
                chosen_user_gpmw = np.random.choice(Users, p=mixed_strategy_adversary_gpmw)
                chosen_user_ucb = np.random.choice(Users, p=mixed_strategy_adversary_ucb)
                chosen_user_stableopt = np.random.choice(Users, p=mixed_strategy_adversary_stableopt)
            else:
                chosen_user = np.random.choice(Users)
                chosen_user_gpmw = chosen_user
                chosen_user_ucb = chosen_user
                chosen_user_stableopt = chosen_user


            """ GPMW player """
            chosen_movie_gpmw = int(np.random.choice(range(N_movies), p=mixed_strategy_GPMW))
            GPMW_payoffs.append( f(chosen_user_gpmw, chosen_movie_gpmw) )

            for movie in range(N_movies):
                GPMW_cum_f[movie] = GPMW_cum_f[movie]  + f(chosen_user_gpmw, movie)

            regret_t = 1/(t + 1) * (np.max(GPMW_cum_f) - np.sum(GPMW_payoffs))
            GPMW_regrets.append(regret_t)

            """ GPUCB player """
            chosen_movie_ucb = np.unravel_index(np.argmax(UCB_gpucb_matrix, axis=None), UCB_gpucb_matrix.shape)[1]

            GPUCB_payoffs.append(f(chosen_user_ucb, chosen_movie_ucb))

            for movie in range(N_movies):
                GPUCB_cum_f[movie] = GPUCB_cum_f[movie] + f(chosen_user_ucb, movie)

            regret_t = 1 / (t + 1) * (np.max(GPUCB_cum_f) - np.sum(GPUCB_payoffs))
            GPUCB_regrets.append(regret_t)

            """ StableOpt player """
            chosen_movie_stableopt = np.argmax( np.min(UCB_stableopt_matrix, axis = 0) )

            StableOpt_payoffs.append(f(chosen_user_stableopt, chosen_movie_stableopt))

            for movie in range(N_movies):
                StableOpt_cum_f[movie] = StableOpt_cum_f[movie] + f(chosen_user_stableopt, movie)

            regret_t = 1 / (t + 1) * (np.max(StableOpt_cum_f) - np.sum(StableOpt_payoffs))
            StableOpt_regrets.append(regret_t)

            """"""""""""""""""""""""""""""""" UPDATES """""""""""""""""""""""""""""""""""

            """ GPMW player """
            idx_user = int( np.where(Users == chosen_user_gpmw)[0] )
            payoffs_hindsight = UCB_gpmw_matrix[idx_user,:]
            payoffs_hindsight = np.minimum( payoffs_hindsight, 5.0*np.ones(N_movies))
            payoffs_hindsight = np.maximum( payoffs_hindsight, 0.0*np.ones(N_movies))
            losses = np.ones(N_movies) - np.array(payoffs_hindsight)/5
            gamma_t = np.sqrt(8*np.log(N_movies) / T)
            mixed_strategy_GPMW = np.multiply(mixed_strategy_GPMW, np.exp(np.multiply(gamma_t, -losses)))
            mixed_strategy_GPMW = mixed_strategy_GPMW / np.sum(mixed_strategy_GPMW)

            X_gpmw[idx_user].append(movies_features[chosen_movie_gpmw, :])
            y_gpmw[idx_user].append(f(chosen_user_gpmw, chosen_movie_gpmw))
            GPs[idx_user].fit(X_gpmw[idx_user], y_gpmw[idx_user])

            payoffs = np.zeros(N_movies)
            for movie in range(N_movies):
                mu, sigma = GPs[idx_user].predict(movies_features[movie, :].reshape(1, -1), return_std=True)
                beta_t = 2.0
                payoffs[movie] = mu + beta_t * sigma
            UCB_gpmw_matrix[idx_user,:] = payoffs

            """ GPUCB player """
            idx_user = int( np.where(Users == chosen_user_ucb)[0] )

            X_ucb[idx_user].append(movies_features[chosen_movie_ucb, :])
            y_ucb[idx_user].append(f(chosen_user_ucb, chosen_movie_ucb))
            GPs[idx_user].fit(X_ucb[idx_user], y_ucb[idx_user])

            payoffs = np.zeros(N_movies)
            for movie in range(N_movies):
                mu, sigma = GPs[idx_user].predict(movies_features[movie, :].reshape(1, -1), return_std=True)
                beta_t = 2.0
                payoffs[movie] = mu + beta_t * sigma
            UCB_gpucb_matrix[idx_user, :] = payoffs

            """ StableOpt player """
            idx_user = np.argmin( LCB_stableopt_matrix[Users, chosen_movie_stableopt])

            X_stableopt[idx_user].append(movies_features[chosen_movie_stableopt, :])
            y_stableopt[idx_user].append(f(chosen_user_stableopt, chosen_movie_stableopt))
            GPs[idx_user].fit(X_stableopt[idx_user], y_stableopt[idx_user])

            payoffs = np.zeros(N_movies)
            lcb_payoffs = np.zeros(N_movies)
            for movie in range(N_movies):
                mu, sigma = GPs[idx_user].predict(movies_features[movie, :].reshape(1, -1), return_std=True)
                beta_t = 2.0
                payoffs[movie] = mu + beta_t * sigma
                lcb_payoffs[movie] = mu - beta_t *sigma
            UCB_stableopt_matrix[idx_user, :] = payoffs
            LCB_stableopt_matrix[idx_user, :] = lcb_payoffs


            if adaptive_adv: # update mixed strategy of adversaries
                gamma_adv_t = np.sqrt(np.log(N_users) / T)
                losses_adv = rating_Matrix[Users, chosen_movie_gpmw].T
                losses_adv = losses_adv/np.max(losses_adv)
                mixed_strategy_adversary_gpmw = np.multiply(mixed_strategy_adversary_gpmw, np.exp(np.multiply(gamma_adv_t, -losses_adv)))
                mixed_strategy_adversary_gpmw = mixed_strategy_adversary_gpmw / np.sum(mixed_strategy_adversary_gpmw)

                losses_adv = rating_Matrix[Users, chosen_movie_ucb].T
                losses_adv = losses_adv/np.max(losses_adv)
                mixed_strategy_adversary_ucb = np.multiply(mixed_strategy_adversary_ucb, np.exp(np.multiply(gamma_adv_t, -losses_adv)))
                mixed_strategy_adversary_ucb = mixed_strategy_adversary_ucb / np.sum(mixed_strategy_adversary_ucb)

                losses_adv = rating_Matrix[Users, chosen_movie_stableopt].T
                losses_adv = losses_adv/np.max(losses_adv)
                mixed_strategy_adversary_stableopt = np.multiply(mixed_strategy_adversary_stableopt, np.exp(np.multiply(gamma_adv_t, -losses_adv)))
                mixed_strategy_adversary_stableopt = mixed_strategy_adversary_stableopt / np.sum(mixed_strategy_adversary_stableopt)


        Regrets_GPMW.append(GPMW_regrets)
        Regrets_GPUCB.append(GPUCB_regrets)
        Regrets_StableOpt.append(StableOpt_regrets)

    with open('saved_Regrets_adaptive_adv'+ str(adaptive_adv)+'.pckl', 'wb') as f:
        pickle.dump((Regrets_GPMW), f)
        pickle.dump((Regrets_GPUCB), f)
        pickle.dump((Regrets_StableOpt), f)

