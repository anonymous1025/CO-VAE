from __future__ import division
from __future__ import print_function

import time
import os
import sys

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import scipy.stats as stats

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

from sklearn import manifold
from scipy.special import expit

from optimizer import OptimizerSiemens,OptimizerVAE
from input_data import *
from model_adj import *
from preprocessing import *

def sigmoid(x):
        return 1 / (1 + np.exp(-x))


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 10, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 2, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden2_2', 1, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('edge_dropout', 0., 'Dropout for individual edges in training graph')
flags.DEFINE_float('autoregressive_scalar', 0., 'Scalar for Graphite')
flags.DEFINE_integer('vae', 1, 'for variational objective')
flags.DEFINE_integer('batch_size', 12, 'Number of samples in a batch.')
flags.DEFINE_integer('decoder_batch_size', 12, 'Number of samples in a batch.')
flags.DEFINE_string('vae_type', 'local', 'local or global or local_global')
flags.DEFINE_integer('subsample', 0, 'Subsample in optimizer')
flags.DEFINE_float('subsample_frac', 1, 'Ratio of sampled non-edges to edges if using subsampling')
flags.DEFINE_integer('num_feature', 16, 'Number of features.')
flags.DEFINE_integer('verbose', 1, 'Output all epoch data')
flags.DEFINE_integer('test_count', 10, 'batch of tests')
flags.DEFINE_string('dataset', 'brain', 'Dataset string.')
flags.DEFINE_string('model', 'feedback', 'Model string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('seeded', 1, 'Set numpy random seed')
flags.DEFINE_integer('connected_split', 1, 'use split with training set always connected')
flags.DEFINE_string('type', 'test', 'train or test')
flags.DEFINE_integer('if_visualize', 1, 'varying the z to see the generated graphs')

def ZscoreNormalization(x, mean_, std_):
    """Z-score normaliaztion"""
    x = (x - mean_) / std_
    return x


def main(beta,type_):
    
        if FLAGS.seeded:
            np.random.seed(1)
        
        dataset_str = FLAGS.dataset
        model_str = FLAGS.model
        
          # Load data
        adj, features = load_data_protein()
        adj_def = adj
        adj_orig = adj
        adj_train=adj[:12000]
        adj_test=adj[:6000]
        feature_train=features[:12000]
        feature_test=features[:6000]
        if FLAGS.features == 0:
              feature_test = np.tile(np.identity(feature_test.shape[1]),[feature_test.shape[0],1,1])
              feature_train = np.tile(np.identity(feature_train.shape[1]),[feature_train.shape[0],1,1])
            # featureless
        num_nodes = adj.shape[1]
        
        #features = sparse_to_tuple(features.tocoo())
        num_features = feature_test.shape[2]
        pos_weight = float(adj.shape[0] *adj.shape[1] * adj.shape[1] - adj.sum()) / adj.sum()
        norm = adj.shape[0] *adj.shape[1] * adj.shape[1] / float((adj.shape[0] *adj.shape[1] * adj.shape[1] - adj.sum()) * 2)
        
        adj_orig=adj_train.copy()
        for i in range(adj_train.shape[0]):
            adj_orig[i] = adj_train[i].copy() + np.eye(adj_train.shape[1])
            
        #use encoded label
        adj_label=np.zeros(([adj_train.shape[0],adj_train.shape[1],adj_train.shape[2],2]))  
        for i in range(adj_train.shape[0]):
                for j in range(adj_train.shape[1]):
                    for k in range(adj_train.shape[2]):
                        adj_label[i][j][k][int(adj_orig[i][j][k])]=1
        
        placeholders = {
                'features': tf.placeholder(tf.float32,[FLAGS.batch_size,feature_train.shape[1],feature_train.shape[2]]),
                'adj': tf.placeholder(tf.float32,[FLAGS.batch_size,adj_train.shape[1],adj_train.shape[2]]),
                'adj_orig': tf.placeholder(tf.float32,[FLAGS.batch_size,adj_train.shape[1],adj_train.shape[2],2]),
                'dropout': tf.placeholder_with_default(0., shape=()),
            }
        
        #model = GCNModelFeedback(placeholders, num_features, num_nodes)
        model = GCNModelVAE(placeholders, num_features, num_nodes)
        
        def generate_new(adj_test,adj_label,features):
           feed_dict = construct_feed_dict(adj_test, adj_label, features, placeholders)
           feed_dict.update({placeholders['dropout']: 0})
           z = sess.run([ model.sample_reconstructions], feed_dict=feed_dict)
           #adj_rec=adj_rec.reshape(-1,76,76)
           # Predict on test set of edges
           #adj_rec = np.dot(emb, np.transpose(emb,[0,2,1]))
           return z
       
        if type_=='train':
          with tf.name_scope('optimizer'):
                opt = OptimizerVAE(preds=model.reconstructions,
                                   labels=tf.reshape(placeholders['adj_orig'], [-1,2]),
                                   model=model, num_nodes=num_nodes,
                                   pos_weight=pos_weight,
                                   norm=norm,
                                   beta=beta)
        
     
        saver = tf.train.Saver()
        if FLAGS.type=='train':
          with tf.Session() as sess:
            sess.run(tf.global_variables_initializer()) 
             # Train model
            for epoch in range(FLAGS.epochs):
              batch_num=int(adj_train.shape[0]/FLAGS.batch_size)
              graph=[]
              for i in range(batch_num):
                  adj_batch=adj_train[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                  adj_label_batch=adj_label[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                  feature_batch=feature_train[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                  t = time.time()
                  # Construct feed dictionary
                  feed_dict = construct_feed_dict(adj_batch, adj_label_batch, feature_batch,placeholders)
                  feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                  # Run single weight update
                  outs = sess.run([opt.opt_op, opt.cost , opt.accuracy, model.z_mean], feed_dict=feed_dict)
                  # Compute average loss
                  avg_cost = outs[1]
                  avg_accuracy = outs[2]
                  if epoch==FLAGS.epochs-1:
                      graph.append(outs[3])
        
                  print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "train_acc=", "{:.5f}".format(avg_accuracy),
                  "time=", "{:.5f}".format(time.time() - t))  
                  if epoch%80==0:
                      save_path = saver.save(sess, "C:/Users/gxjco/Desktop/protein/disentangle_evaluation/tmp/model_dgt_"+str(epoch)+'_'+str(beta)+".ckpt")
            print("Optimization Finished!")
            print("Model saved in file: %s" % save_path)
            
        if type_=='test':
          with tf.Session() as sess:
            saver.restore(sess, "C:/Users/gxjco/Desktop/protein/disentangle_evaluation/tmp/model_dgt_"+str(80)+'_'+str(beta)+".ckpt")
            print("Model restored.")
            graphs=[]
            z=[]
            test_batch_num=int(adj_train.shape[0]/FLAGS.batch_size)
            for i in range(10):
                i=1
                adj_batch_test=adj_train[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                adj_batch_label=adj_label[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                feature_batch_test=feature_train[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                z_mean=generate_new(adj_batch_test,adj_batch_label,feature_batch_test)
                #graphs.append(graph)
                z.append(z_mean)
            #graphs=np.array(graphs).reshape(12000,76,76)-np.tile(np.eye(76),[12000,1,1])
            #np.save('graphs_dgt.npy',graphs)
            
            graphs=np.array(z)
            np.save('z_dgt'+'_'+str(beta)+'_node60.npy',z)
            '''
            graphs=np.array(graphs).reshape(6000,76,76)
            def find_nodes(mat):
              node=[]
              for i in range(76):
                for j in range(76):
                    if mat[i][j]==1:
                        if i+1 not in node: node.append(i+1)
                        if j+1 not in node: node.append(j+1)
              return node
        
            G = open('generated_graphs_graphite_dgt.txt', 'w')
            for i in range(len(graphs)):
              G.write('BEGIN MODEL '+str(i)+'\n')
              node_list=find_nodes(graphs[i])
              G.write('nodes:'+ str(node_list)[1:-1]+'\n')  
              for m in range(76):
                  for n in range(76):
                      if graphs[i][m][n]==1:
                          G.write(str(m+1)+' '+str(n+1)+'\n')
            G.close()    
        '''
if __name__ == '__main__':
      for beta in [250]:
         tf.reset_default_graph()
         main(beta,'test')