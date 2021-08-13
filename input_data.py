import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import scipy.io as io



def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_protein():
    n = io.loadmat("data/Homo_sapiens.mat")
    return n['network'], n['group']

def load_enzyme():
    adj = sp.lil_matrix((125, 125))
    features = sp.lil_matrix((125, 1))
    for line in open("data/ENZYMES_g296.edges"):
        vals = line.split()
        x = int(vals[0]) - 2
        y = int(vals[1]) - 2
        adj[y, x] = adj[x, y] = 1
    return adj, features

def load_florida():
    adj = sp.lil_matrix((128, 128))
    features = sp.lil_matrix((128, 1))
    for line in open("data/eco-florida.edges"):
        vals = line.split()
        x = int(vals[0]) - 1
        y = int(vals[1]) - 1
        val = float(vals[2])
        adj[y, x] = adj[x, y] = val
    return adj, features

def load_brain():
    adj = sp.lil_matrix((1780, 1780))
    features = sp.lil_matrix((1780, 1))
    nums = []
    for line in open("data/bn-fly-drosophila_medulla_1.edges"):
        vals = line.split()
        x = int(vals[0]) - 1
        y = int(vals[1]) - 1
        adj[y, x] = adj[x, y] = adj[x, y] + 1
    return adj, features


def load_data(dataset):
    if dataset == 'florida':
        return load_florida()
    elif dataset == 'brain':
        return load_brain()
    elif dataset == 'enzyme':
        return load_enzyme()
    elif dataset == 'protein':
        return load_protein()

    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    # if dataset == 'cora':
    #     names = ['y', 'ty', 'ally']
    #     objects = []
    #     for i in range(len(names)):
    #         objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
    #     y, ty, ally = tuple(objects)

    #     labels = np.vstack((ally, ty))
    #     labels[test_idx_reorder, :] = labels[test_idx_range, :]
    #     np.save('labels', labels)


    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features

def load_data_protein():
    
    def find_edge(list_e): 
        edges=[]
        for e in list_e:
            e1=e.split(' ')[0].split('-')[1]
            e2=e.split(' ')[1].split('-')[1].split('\n')[0]
            edges.append((int(e1),int(e2)))
        return edges 
    def convert_to_one_hot(y, C):
       return np.eye(C)[y.reshape(-1)]
   
    graph=[]
    n=0
    with open ('C:/Users/gxjco/Desktop/protein/1DTJA_near_natives_ids_letters.txt') as f:
        for line in f:
            if line[0:2]=='Se': node=line
            if line[0]=='N' and line[1]!='-':
                edge=[]
                edge.append(line)
                graph.append(edge)
                n=n+1
            if line[1]=='-':
                graph[n-1].append(line)   
        
    adj=[]        
    for g in graph:
        G=nx.Graph()
        G.add_nodes_from(list(range(1,77)))
        G.add_edges_from(find_edge(g[1:]))
        adj.append(nx.to_numpy_matrix(G))
        
    n=node.split(' ')[1].split(',')
    for i in range(len(n)):
       n[i]=n[i][0]
    node_type=list(set(n))    
    
    feature=np.zeros((76,16))
    for i in range(76):
        type_idx=node_type.index(n[i])
        feature[i][type_idx]=1
        
    return np.array(adj), np.tile(feature,[len(adj),1,1])

