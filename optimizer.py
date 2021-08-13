import tensorflow as tf
from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm,beta):
        preds_sub = preds
        labels_sub = labels

        #self.cost = norm * tf.reduce_mean(tf.squared_difference(labels_sub,preds_sub))
        self.cost=norm*tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)
        self.log_lik = self.cost

        if FLAGS.vae_type=='local_global':
            self.kl_n = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std_n - tf.square(model.z_mean_n) - tf.square(tf.exp(model.z_log_std_n)), 1))
            self.kl_g = 0.5 * tf.reduce_mean(1 + 2 * model.z_log_std_g - tf.square(model.z_mean_g) - tf.square(tf.exp(model.z_log_std_g)))         
            self.cost -= 250(self.kl_n+self.kl_g)
            
        if FLAGS.vae_type=='local':
            self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) - tf.square(tf.exp(model.z_log_std)), 1))
            self.cost -= beta*self.kl  
            
        if FLAGS.vae_type=='global':
            self.kl = 0.5 * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std_g - tf.square(model.z_mean_g) - tf.square(tf.exp(model.z_log_std_g)), 1))
            self.cost -= 250*self.kl     


        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32), tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

class OptimizerSiemens(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        neg_cost = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=0))
        pos_cost = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=1)) - neg_cost
        total = tf.reduce_sum(labels)

        self.cost = pos_cost / total + neg_cost / (num_nodes * num_nodes - total)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.log_lik = self.cost

        if FLAGS.vae:
            self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std_g - tf.square(model.z_mean_g) - tf.square(tf.exp(model.z_log_std_g)), 1))
            self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32), tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
