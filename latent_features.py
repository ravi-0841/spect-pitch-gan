import tensorflow as tf
import scipy.io as scio
import pylab
import numpy as np
import os
import time

import utils.preprocess as preproc

import modules.base_modules as base_mods
from utils.model_utils import l1_loss


def shuffle_feats_label(features, label):
    
    assert features.shape[0]==label.shape[0]
    shuffle_ids = np.arange(0, features.shape[0])
    np.random.shuffle(shuffle_ids)
    return features[shuffle_ids], label[shuffle_ids]


def classifier_model(input_mfc, reuse=False, scope_name='classifier'):
    
    # expects input_mfc of shape [#batch #dim_mfc, #timesteps]
    
    input_mfc_transposed = tf.transpose(input_mfc, [0,2,1],
                                        name='classifier_input_transpose')
    
    with tf.variable_scope(scope_name) as scope:
        
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False
        
        h1_conv = base_mods.conv1d_layer(inputs=input_mfc_transposed, filters=64, 
                               kernel_size=15, strides=1, activation=None, 
                               name='h1_conv')
        h1_gates = base_mods.conv1d_layer(inputs=input_mfc_transposed, filters=64, 
                                kernel_size=15, strides=1, activation=None, 
                                name='h1_gates')
        h1_glu = base_mods.gated_linear_layer(inputs=h1_conv, gates=h1_gates, 
                                    name='h1_glu')
        
        d1 = base_mods.downsample1d_block(inputs=h1_glu, filters=128, 
                                kernel_size=5, strides=2, 
                                name_prefix='downsample_1_')

        r1 = base_mods.residual1d_block(inputs=d1, filters=256, \
                kernel_size=3, strides=1, \
                name_prefix='residual1d_block1_')

        u1 = base_mods.upsample1d_block(inputs=r1, filters=256, \
                kernel_size=5, strides=1, \
                shuffle_size=2, name_prefix='upsample1d_block1_')
        
        o1 = base_mods.conv1d_layer(inputs=u1, filters=1, 
                                    kernel_size=15, strides=1, 
                                    activation=None, name='latent_feats_')
        
        o2 = tf.layers.dense(inputs=o1, units=1, \
                             activation=tf.nn.sigmoid)

        return o1, tf.reduce_mean(tf.squeeze(o2, axis=-1), axis=-1, 
                                  keep_dims=True)


class CNN_classifier():
    
    def __init__(self, dim_mfc=23, classifier=classifier_model, pre_train=None):
        
        self.mfc_shape = [None, dim_mfc, None]
        self.label_shape = [None, 1]
        
        self.classifier = classifier
        
        self.build_model()
        self.optimizer_initializer()
        
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        
        if pre_train is not None:
            self.saver.restore(self.sess, pre_train)
        else:
            self.sess.run(tf.global_variables_initializer())
            

    def build_model(self):
        
        self.features = tf.placeholder(tf.float32, shape=self.mfc_shape, 
                                       name='features')
        
        self.labels = tf.placeholder(tf.float32, shape=self.label_shape, 
                                     name='labels')
        
        # features for embedding generation
        self.features_embed = tf.placeholder(tf.float32, shape=self.mfc_shape, 
                                             name='features')
        
        # get latent embedding and prediction
        self.latent_embedding, self.prediction \
            = self.classifier(input_mfc=self.features, reuse=False, 
                              scope_name='classifier')
        
        self.classifier_loss = l1_loss(y=self.prediction, y_hat=self.labels)
    
        variables = tf.trainable_variables()
        self.classifier_vars = [var for var in variables if 'classifier' in var.name]

        self.generate_embedding, self.generate_prediction \
            = self.classifier(input_mfc=self.features_embed, reuse=True)

        
    def optimizer_initializer(self):
        
        self.learning_rate = tf.placeholder(tf.float32, None, 
                                            name='learning_rate')
        self.classifier_optimizer \
            = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                                     beta1=0.5).minimize(self.classifier_loss, 
                                              var_list=self.classifier_vars)

        
    def train(self, input_mfcs, labels, learning_rate):

        loss, embedding, prediction, _ = self.sess.run([self.classifier_loss, 
                                            self.latent_embedding, 
                                            self.prediction, 
                                            self.classifier_optimizer], 
                                            feed_dict={self.features:input_mfcs, 
                                                       self.labels:labels, 
                                                       self.learning_rate:learning_rate})
    
        return loss, embedding, prediction


    def get_embedding(self, input_mfcs):
        
        embeddings = self.sess.run(self.generate_embedding, 
                                   feed_dict={self.features_embed:input_mfcs})
        return embeddings
    
    def get_predictions(self, input_mfcs):
        
        predictions = self.sess.run(self.generate_prediction, 
                                   feed_dict={self.features_embed:input_mfcs})
        return predictions

            
    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))


    def load(self, filepath):

        self.saver.restore(self.sess, filepath)


if __name__ == '__main__':
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    data = scio.loadmat('./data/neu-ang/train_mod_dtw_harvest.mat')
    
    mfc_A = np.vstack(np.transpose(data['src_mfc_feat'], (0,1,3,2)))
    mfc_B = np.vstack(np.transpose(data['tar_mfc_feat'], (0,1,3,2)))

    mfc_feats = np.concatenate((mfc_A, mfc_B), axis=0)    
    labels = np.concatenate((np.zeros((mfc_A.shape[0],1)), 
                             np.ones((mfc_B.shape[0],1))), axis=0)
    
    mini_batch_size = 256
    learning_rate = 1e-03
    num_epochs = 100
    
    model = CNN_classifier(dim_mfc=23, pre_train=None)
    
    loss_log = list()
    for epoch in range(1,num_epochs+1):

        print('Epoch: %d' % epoch)

        start_time_epoch = time.time()
        n_samples = mfc_feats.shape[0]
        
        mfc_feats, labels = shuffle_feats_label(mfc_feats, labels)
        
        train_loss = list()

        for i in range(n_samples // mini_batch_size):

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            loss, embed, predict = model.train(input_mfcs=mfc_feats[start:end], 
                                               labels=labels[start:end], 
                                               learning_rate=learning_rate)
            
            train_loss.append(loss)
        
        loss_log.append(np.mean(train_loss))
        print('Loss in epoch %d- %f' % (epoch, np.mean(train_loss)))
        
        model.save(directory='./model', filename='embedding_net.ckpt')
        
        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, \
                (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))














