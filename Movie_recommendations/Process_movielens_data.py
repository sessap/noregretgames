import numpy as np
from sklearn.decomposition import NMF
import pickle

N_users = 943
N_movies = 1682

rating_Matrix_original = np.zeros((N_users, N_movies))

loaded_data = np.loadtxt('ml-100k/u.data')

for d in range( np.size(loaded_data,0)):
    user     = int( loaded_data[d,0]-1)
    movie    = int( loaded_data[d,1]-1)
    rating   = loaded_data[d,2]
    rating_Matrix_original[user,movie] = rating


nmf = NMF(15)  #NMF with 15 components
users_features = nmf.fit_transform(rating_Matrix_original)
movies_features = nmf.components_

rating_Matrix = np.dot(users_features, movies_features)

with open('rating_features_and_matrix.pckl', 'wb') as f:
    pickle.dump((rating_Matrix), f)
    pickle.dump((users_features), f)
    pickle.dump((movies_features), f)