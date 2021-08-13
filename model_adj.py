# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:27:32 2019

@author: gxjco
"""

from layers import GraphConvolution_adj,n2g_adj,de_n2g,e2e,e2n,de_e2e,de_e2n,lrelu,batch_norm
from layers import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class GCNModelVAE(object):
        

    '''VGAE Model for reconstructing graph edges from node representations.'''
    def __init__(self, placeholders, num_features, num_nodes, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = num_features
        #self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adj_label = tf.reshape(placeholders['adj_orig'],[-1,2])
        self.weight_norm = 0
        
        self.g_bn_e1 = batch_norm(name='g_bn_e1')
        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
 

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
    

        self._build()
        
    def _build(self):

        if FLAGS.type=='train':
           self.encoder(self.adj)
           z = self.get_z(random = True)
           self.reconstructions = self.decoder(z)
        if FLAGS.type=='test': 
          self.sample_reconstructions= self.sample()
          #self.encoder(sample_reconstructions)
        t_vars = tf.trainable_variables()

        self.vars = [var for var in t_vars]

        self.saver = tf.train.Saver()
        #self.reconstructions_noiseless = self.decoder(z_noiseless)
     
    def encoder(self,adj):
      with tf.variable_scope("encoder") as scope:  
        adj=tf.cast(tf.reshape(adj,[FLAGS.batch_size,self.n_samples,self.n_samples,1]),dtype=tf.float32) 
        e1 = self.g_bn_e1(e2e(lrelu(adj), FLAGS.hidden1, k_h=self.n_samples,name='g_e1_conv'))
            # e1 is (n*300 x 300*d )
        e2 = self.g_bn_e2(e2e(lrelu(e1), FLAGS.hidden2, k_h=self.n_samples,name='g_e2_conv'))
        #self.z_mean=self.g_bn_e3(e2n(lrelu(e2), FLAGS.hidden2,k_h=self.n_samples, name='g_e3_conv'))    
        
        #self.z_log_std=self.g_bn_e4(e2n(lrelu(e2), FLAGS.hidden2,k_h=self.n_samples, name='g_e4_conv')) 
        
        #hidden=self.g_bn_e1(GraphConvolution_adj(self.adj,self.inputs,FLAGS.hidden1,name='g_e1_conv'))
        
        self.z_mean=self.g_bn_e3(GraphConvolution_adj(lrelu(e2),self.inputs, FLAGS.hidden2_2, name='g_e3_conv'))    
        
        self.z_log_std=self.g_bn_e4(GraphConvolution_adj(lrelu(e2),self.inputs, FLAGS.hidden2_2, name='g_e4_conv'))             
 

    def get_z(self, random):
        z = self.z_mean+ tf.random_normal([FLAGS.batch_size,self.n_samples,1,FLAGS.hidden2*FLAGS.hidden2_2],stddev=0.1) * tf.exp(self.z_log_std)
        if not random or not FLAGS.vae:
          z = self.z_mean
        return z


    def decoder(self, z):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as scope: 
                        
            #z=tf.reshape(z,[FLAGS.decoder_batch_size,self.n_samples,1,FLAGS.hidden2])
            
            #d1=de_n2g(tf.nn.relu(z),
             #   [FLAGS.decoder_batch_size, 76, 1, FLAGS.hidden2],k_h=76, name='g_d1', with_w=False)
            
            #d1_ = self.g_bn_d2(d1)
            
            d1= de_e2n(tf.nn.relu(z),
                [FLAGS.decoder_batch_size, self.n_samples, self.n_samples, FLAGS.hidden2*FLAGS.hidden2_2],k_h=self.n_samples, name='g_d1', with_w=False)
            
            d1_ = self.g_bn_d1(d1)
 
            d2= de_e2e(tf.nn.relu(d1_),
                [FLAGS.decoder_batch_size,self.n_samples, self.n_samples, FLAGS.hidden1],k_h=self.n_samples, name='g_d2', with_w=False)
            
            d2_ = self.g_bn_d2(d2)
       
            d3= de_e2e(tf.nn.relu(d2_),
                [FLAGS.decoder_batch_size, self.n_samples, self.n_samples, 2],k_h=self.n_samples, name='g_d3', with_w=False)
            

            return  tf.reshape(d3,[-1, 2])




    def sample(self):
        #z = np.random.normal(size=[FLAGS.batch_size,self.n_samples,1,FLAGS.hidden2*FLAGS.hidden2_2],scale=0.1).astype(np.float32)
        #z=np.random.normal(size=[FLAGS.batch_size,,FLAGS.hidden2],scale=10).astype(np.float32)
        if FLAGS.if_visualize:
            
            #make one dimension of one node changed and other fixed
            z = np.random.normal(size=[1,self.n_samples,FLAGS.hidden2*FLAGS.hidden2_2],scale=0.1).astype(np.float32)
            z_orig = np.tile(z,[FLAGS.batch_size,1,1])
            rang= [2000,10000,20000,100000,200000,1000000]
            for fix_dim in range(FLAGS.hidden2*FLAGS.hidden2_2): 
              for i in range(fix_dim*6,fix_dim*6+6):
                  z_orig[i,60,fix_dim]=rang[int(i%6)] 
                                    
            #make one dimension fixed and other changed
            #z = np.random.normal(size=[FLAGS.batch_size,self.n_samples,FLAGS.hidden2*FLAGS.hidden2_2],scale=0.1).astype(np.float32)
            #for fix_dim in range(FLAGS.hidden2*FLAGS.hidden2_2-1): 
            #  for i in range(fix_dim*100,fix_dim*100+100):
             #     z[i,:,fix_dim]= z[fix_dim*100,:,fix_dim].copy()
             
        z=z_orig.reshape([FLAGS.batch_size,self.n_samples,1,FLAGS.hidden2*FLAGS.hidden2_2])
            
        reconstruction= self.decoder(z)
        reconstruction = tf.sigmoid(tf.reshape(reconstruction, [FLAGS.batch_size,self.n_samples, self.n_samples,2]))
        reconstruction=tf.arg_max(reconstruction,3)-np.tile(np.eye(76),[FLAGS.batch_size,1,1])
      
        return reconstruction


