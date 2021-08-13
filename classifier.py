# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:22:32 2019

@author: gxjco

process data and classify the factors
"""

import numpy as np



def compute_var_dim(x,d):
     for m in range(76):
        for n in range(d):
              x[:,m,n]=x[:,m,n]/np.std(x[:,m,n]) #normlaize each dimension by sdt over full data
     var=np.ones((76,d))
     for m in range(76):
        for n in range(d): 
          var[m,n]=np.var(x[:,m,n])
     var=np.mean(var,axis=0)  #compute the variance of each factor
     min_var=np.argmin(var)
     return min_var

def metric(beta,k_n): 
  d_values=[]
  k_values=[]   
  z=np.load('z_dgt'+'_'+str(beta)+'.npy').reshape(1000,k_n*10,76,k_n) 
  for k in range(k_n): 
    for i in range(1000):
       x=z[i,k*10:k*10+10]
       d=compute_var_dim(x,k_n)
       d_values.append(d)
       k_values.append(k)

  #classify the results
  v_matrix=np.zeros((k_n,k_n))  
  for j in range(k_n):
    for k in range(k_n):
      v_matrix[j,k]=np.sum((np.array(d_values)==j)&(np.array(k_values)==k))   
    
  classifier=np.argmax(v_matrix,axis=1)
  predicted_k=classifier[d_values]
  acc=np.sum(predicted_k==k_values)/len(predicted_k)
  return acc  

for beta in [2000]:#,1000,500,250,100,50,10,1
   print(metric(beta,16)) 