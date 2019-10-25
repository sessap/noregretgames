import pandas
import networkx as nx
import itertools
import numpy as np
import matplotlib.pyplot as plt


def k_shortest_paths(G, source, target, k, weight=None):
    return list(itertools.islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

class NetworkData:
     def __init__(self, Nodes, Edges, Capacities, Freeflowtimes, Powers):
        self.Nodes   = np.reshape(Nodes, (-1,1) )
        self.Edges   = np.reshape(Edges, (-1,2) ) 
        self.Capacities   = np.reshape(Capacities, (-1,1))
        self.Freeflowtimes     = np.reshape(Freeflowtimes   , (-1,1))
        self.Powers        = np.reshape(Powers, (-1,1))
    
    
## Create Sioux-Falls Network
def Create_Network():
    SiouxNetwork = nx.DiGraph()

    reader = pandas.read_csv("SiouxFalls_node.csv")
    x_coords = reader['X'].values
    y_coords = reader['Y'].values
    for num in range(24):
        SiouxNetwork.add_node(str(num+1) , pos = (x_coords[num], y_coords[num]) )

    reader = pandas.read_csv("SiouxFalls_net.csv")
    init_nodes  = reader['Init_node'].values
    term_nodes  = reader['Term_node'].values
    lengths     = reader['Length'].values

    for e in range(len(init_nodes)):
        SiouxNetwork.add_edge(str(init_nodes[e]), str(term_nodes[e]) , weight = lengths[e])

    nx.draw(SiouxNetwork , nx.get_node_attributes(SiouxNetwork, 'pos') , with_labels=True, node_size=500)
    nx.draw_networkx_edge_labels(SiouxNetwork , nx.get_node_attributes(SiouxNetwork, 'pos') , edge_labels = nx.get_edge_attributes(SiouxNetwork,'weight'))

    Nodes = np.arange(1, 25)
    Edges = np.array([init_nodes, term_nodes]).T
    Capacities = reader['Capacity'].values / 100
    Freeflowtimes = reader['Free_Flow_Time'].values
    Powers = reader['Power'].values
    SiouxNetwork_data = NetworkData(Nodes, Edges, Capacities, Freeflowtimes, Powers)

    return   SiouxNetwork, SiouxNetwork_data

def get_edge_idx (Edges, node1, node2):
    idx = np.where(np.all(Edges == [int(node1), int(node2)] ,axis=1))
    return idx

def Compute_Strategy_vectors(OD_pairs, Demands, Freeflowtimes, Networkx, Edges):
    E = len(Edges)
    K = 5  # K shortest paths for each agent
    Strategy_vectors = [()]*len(OD_pairs)
    for i in range(len(OD_pairs)):
        Strategy_vectors[i] = list()
        OD_pair = np.array(OD_pairs[i])
        paths = k_shortest_paths(Networkx, str(OD_pair[0]), str(OD_pair[1]), K, weight = 'weight')
        for a in range(len(paths)):
            vec = np.zeros((E,1))
            for n in range(len(paths[a])-1):
                idx = get_edge_idx(Edges,  paths[a][n], paths[a][n+1])
                vec[idx] = 1
            strategy_vec = np.multiply(vec, Demands[i])
            if a == 0:
                Strategy_vectors[i].append(  strategy_vec )
            if a > 0 and np.dot(strategy_vec.T, Freeflowtimes) < 3* np.dot(Strategy_vectors[i][0].T, Freeflowtimes )  :
                Strategy_vectors[i].append(strategy_vec )

    return Strategy_vectors 


def Compute_traveltimes(NetworkData, Strategy_vectors, played_actions, player_id ):
    N = len(Strategy_vectors) # number of players
    Total_occupancies = np.sum([Strategy_vectors[i][played_actions[i]] for i in range(N)], axis = 0)
    
    E = np.size(NetworkData.Edges,0)
    a = NetworkData.Freeflowtimes
    b = np.divide( np.multiply(NetworkData.Freeflowtimes, 0.15*np.ones((E,1))) , np.power(NetworkData.Capacities, NetworkData.Powers))
    congestions = a + np.multiply(b, np.power(Total_occupancies, NetworkData.Powers) )
    if player_id == 'all':
        Traveltimes = np.zeros(N)
        for i in range(N):
            X_i = np.array(Strategy_vectors[i][played_actions[i]])
            Traveltimes[i] = np.dot(X_i.T, congestions )   
        
    else:
        X_i = Strategy_vectors[player_id][played_actions[player_id]]
        Traveltimes = np.dot(X_i.T, congestions )
    
    return Traveltimes
    
