# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:58:20 2019

@author: gxjco
"""
from pylab import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

g=np.load("z_dgt_250_node60.npy").reshape(10,12,76,76)[1]


'''
figure(figsize=(20,100), dpi=200) #
plt.rc('font',family='Times New Roman')
k=0
for i in range(6):
    subplot(6,1,i+1)
    G=nx.from_numpy_matrix(g[k*6+i])  
    nx.draw(G,with_labels=True,pos=nx.circular_layout(G),node_color = 'b',edge_color = 'r',font_size =18,node_size =1500) 
plt.show()         


#evaluate the change of some metrics
def feature(adj):
  G=nx.from_numpy_matrix(adj)
  f=np.zeros(5)
  f[0]=len(G.edges)
  f[1]=nx.density(G)
  f[2]=nx.degree_pearson_correlation_coefficient(G)
  f[3]=nx.transitivity(G)
  return f

edge=[]
density=[]
degree=[]
trans=[]
k=0
for i in range(6):
    f=feature(g[k*6+i])
    edge.append(f[0])
    density.append(f[1])
    degree.append(f[2])
    trans.append(f[3]) 
    
'''    
i=0
graphs = open('graphs_0.txt', 'w')
graphs.write('PFRMAT RR'+'\n')
graphs.write('SEQRES MKELVEMAVPENLVGAILGKGGKTLVEYQELTGARIQISKKGEFLPGTRNRRVTITGSPAATQAAQYLISQRVTYE'+'\n')
graphs.write('MODEL '+str(i)+'\n')
for m in range(76):
        for n in range(76):
            if g[i][m][n]==1:
                graphs.write(str(m+1)+' '+str(n+1)+' '+str(0)+' '+str(8)+' '+str(1.0000)+'\n')
graphs.write('END'+'\n')                
graphs.close()  
