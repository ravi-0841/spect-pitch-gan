import tensorflow as tf
import numpy as np
import os
import scipy.io as scio
import time
import sys


import modules.base_modules_default_init as basic_blocks

from utils.model_utils import l1_loss
from utils.feat_utils import shuffle_feats_label


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def vae_encoder(input_mfc, reuse=False, scope_name='encoder'):
    
    # input is expected to be of shape [#batch, dim_mfc, timesteps]
    input_mfc_transposed = tf.transpose(input_mfc, [0,2,1], 
            name='input_transpose')

    with tf.variable_scope(scope_name) as scope:

        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1_conv = basic_blocks.conv1d_layer(inputs=input_mfc_transposed, 
                filters=64, kernel_size=15, strides=1, activation=None, 
                name='mfc_conv_1d')
        h1_gate = basic_blocks.conv1d_layer(inputs=input_mfc_transposed, 
                filters=64, kernel_size=15, strides=1, activation=None, 
                name='mfc_gate_1d')
        h1_glu = basic_blocks.gated_linear_layer(inputs=h1_conv, gates=h1_gate, 
                name='mfc_glu')

        d1 = basic_blocks.downsample1d_block(inputs=h1_glu, filters=128, 
                kernel_size=5, strides=2, name_prefix='downsample_1')
        d2 = basic_blocks.downsample1d_block(inputs=d1, filters=256, 
                kernel_size=5, strides=2, name_prefix='downsample_2')

        r1 = basic_blocks.residual1d_block(inputs=d2, filters=512, 
                kernel_size=3, strides=1, name_prefix='residual_1')

        u1 = basic_blocks.upsample1d_block(inputs=r1, filters=512, 
                kernel_size=5, strides=1, shuffle_size=2, name_prefix='upsample_1')
        u2 = basic_blocks.upsample1d_block(inputs=u1, filters=256, 
                kernel_size=5, strides=1, shuffle_size=2, name_prefix='upsample_2')

        o1 = basic_blocks.conv1d_layer(inputs=u2, filters=2, 
                kernel_size=15, strides=1, activation=None, name='mean_var')
        o2 = tf.reduce_mean(o1, axis=-1, keep_dims=True, name='classifier_input')

        classifier_branch = tf.layers.dense(inputs=o2, units=1, 
                activation=tf.nn.sigmoid, name='classifier_output')

        mean, log_var = o1[:,:,0:1], o1[:,:,1:2]

        return tf.transpose(mean, [0,2,1]), tf.transpose(log_var, [0,2,1]), \
                tf.reduce_mean(tf.squeeze(classifier_branch, axis=-1), axis=-1, 
                                    keep_dims=True, name='classifer_average')


def vae_decoder(input_noise, reuse=False, final_filters=23, scope_name='decoder'):
    
    # expects the noise to be of shape [#batch, dim_noise, timesteps]
    input_noise_transposed = tf.transpose(input_noise, [0,2,1], 'input_noise_transpose')

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1_conv = basic_blocks.conv1d_layer(inputs=input_noise_transposed, 
                filters=64, kernel_size=15, strides=1, activation=None, 
                name='mfc_conv_1d')
        h1_gate = basic_blocks.conv1d_layer(inputs=input_noise_transposed, 
                filters=64, kernel_size=15, strides=1, activation=None, 
                name='mfc_gate_1d')
        h1_glu = basic_blocks.gated_linear_layer(inputs=h1_conv, gates=h1_gate, 
                name='mfc_glu')

        d1 = basic_blocks.downsample1d_block(inputs=h1_glu, filters=128, 
                kernel_size=5, strides=2, name_prefix='downsample_1')
        d2 = basic_blocks.downsample1d_block(inputs=d1, filters=256, 
                kernel_size=5, strides=2, name_prefix='downsample_2')

        r1 = basic_blocks.residual1d_block(inputs=d2, filters=512, 
                kernel_size=3, strides=1, name_prefix='residual_1')
        r2 = basic_blocks.residual1d_block(inputs=r1, filters=512, 
                kernel_size=3, strides=1, name_prefix='residual_2')
        r3 = basic_blocks.residual1d_block(inputs=r2, filters=512, 
                kernel_size=3, strides=1, name_prefix='residual_3')

        u1 = basic_blocks.upsample1d_block(inputs=r3, filters=512, 
                kernel_size=5, strides=1, shuffle_size=2, name_prefix='upsample_1')
        u2 = basic_blocks.upsample1d_block(inputs=u1, filters=256, 
                kernel_size=5, strides=1, shuffle_size=2, name_prefix='upsample_2')

        o1 = basic_blocks.conv1d_layer(inputs=u2, filters=final_filters, 
                kernel_size=15, strides=1, activation=None, name='mfc_output')

        return tf.transpose(o1, [0,2,1], name='output_transpose')


class VAE(object):

    def __init__(self, encoder=vae_encoder, decoder=vae_decoder, dim_mfc=23, 
            pre_train=None):

        self.mfc_shape = [None, dim_mfc, None]
        self.label_shape = [None, 1]
        self.noise_shape = [None, 1, None]

        self.encoder = encoder
        self.decoder = decoder

        self.model_initializer()
        self.optimizer_initializer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        if pre_train is not None:
            self.saver.restore(self.sess, pre_train)
        else:
            self.sess.run(tf.global_variables_initializer())


    def model_initializer(self):

        # mfc_creating
        self.input_mfc = tf.placeholder(tf.float32, self.mfc_shape, name='input_mfcc')
        self.label = tf.placeholder(tf.float32, self.label_shape, name='labels')
        
        # mfc_for testing
        self.test_mfc = tf.placeholder(tf.float32, self.mfc_shape, name='test_mfcc')
        self.test_embedding = tf.placeholder(tf.float32, self.noise_shape, name='test_embedding')

        # hyperparams
        self.lambda_ae = tf.placeholder(tf.float32, None, name='lambda_ae')
        self.lambda_kl = tf.placeholder(tf.float32, None, name='lambda_KL')


        # generate embedding and get reconstruction from AE
        self.embedding_mean, self.embedding_log_var, self.prediction \
                = self.encoder(input_mfc=self.input_mfc, reuse=False, scope_name='encoder')
        self.embedding_std = tf.math.exp(0.5 * self.embedding_log_var, name='compute_std')
        self.embedding = self.embedding_mean + tf.multiply(self.embedding_std, 
                tf.random.normal(tf.shape(self.embedding_mean), name='embedding_generate'))

        self.reconstruction = self.decoder(input_noise=self.embedding, reuse=False, 
                scope_name='decoder')

        # Compute the AE and classification loss before reshaping
        self.ae_loss = l1_loss(y=self.input_mfc, y_hat=self.reconstruction)
        self.classification_loss = l1_loss(y=self.label, y_hat=self.prediction)
        
        # squeeze the mean and var to compute KL loss
        self.embedding_mean = tf.squeeze(self.embedding_mean)
        self.embedding_log_var = tf.squeeze(self.embedding_log_var)
        self.embedding_std = tf.squeeze(self.embedding_std)
        
        # Compute the KL divergence loss, not really sure if its correct """seems like an appx"""
        self.kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + self.embedding_log_var - tf.pow(self.embedding_mean, 
            2) - tf.math.exp(self.embedding_log_var), axis=1))

        # Compute full VAE loss + classification loss
        self.vae_class_loss = self.classification_loss \
            + self.lambda_ae*self.ae_loss + self.lambda_kl*self.kl_loss
        
        # get variables for optimization
        variables = tf.trainable_variables()
        self.trainable_vars = [var for var in variables if 'encoder' in var.name \
                                                        or 'decoder' in var.name]

        # generating embedding
        self.test_embedding_mean, self.test_embedding_log_var, self.test_prediction \
                = self.encoder(input_mfc=self.test_mfc, reuse=True)
        self.test_embedding_std = tf.math.exp(0.5 * self.test_embedding_log_var, 
                name='compute_test_log_var')

        self.test_gen_embedding = self.test_embedding_mean \
                + tf.multiply(tf.random.normal(tf.shape(self.test_embedding_mean), 
                    name='test_embedding_generate'), self.test_embedding_std)

        self.test_gen_mfc = self.decoder(input_noise=self.test_embedding, reuse=True)


    def optimizer_initializer(self):
        
        self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                beta1=0.5).minimize(self.vae_class_loss, var_list=self.trainable_vars)


    def train(self, mfc_features, labels, lambda_ae, lambda_kl, learning_rate):

        embeddings, class_loss, ae_loss, kl_loss, predictions, _ = self.sess.run([self.embedding, 
            self.classification_loss, self.ae_loss, self.kl_loss, 
            self.prediction, self.optimizer], 
            feed_dict={self.input_mfc:mfc_features, 
                self.label:labels, self.lambda_ae:lambda_ae,
                self.lambda_kl:lambda_kl, 
                self.learning_rate:learning_rate})

        return class_loss, ae_loss, kl_loss, embeddings, predictions


    def get_embedding(self, mfc_features):
        
        embeddings = self.sess.run(self.test_gen_embedding, 
            feed_dict={self.test_mfc:mfc_features})

        return embeddings

    
    def get_mfcc(self, embeddings):

        mfcc_features = self.sess.run(self.test_gen_mfc, 
            feed_dict={self.test_embedding:embeddings})

        return mfcc_features


    def get_prediction(self, mfc_features):

        prediction = self.sess.run(self.test_prediction, 
            feed_dict={self.test_mfc:mfc_features})

        return prediction


    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)

        self.saver.save(self.sess, os.path.join(directory, filename))


if __name__ == '__main__':

    data = scio.loadmat('./data/neu-ang/train_mod_dtw_harvest.mat')
    
    mfc_A = np.vstack(np.transpose(data['src_mfc_feat'], (0,1,3,2)))
    mfc_B = np.vstack(np.transpose(data['tar_mfc_feat'], (0,1,3,2)))

    mfc_feats = np.concatenate((mfc_A, mfc_B), axis=0)    
    labels = np.concatenate((np.zeros((mfc_A.shape[0],1)), 
                             np.ones((mfc_B.shape[0],1))), axis=0)
    
    mini_batch_size = 256
    learning_rate = 1e-03
    num_epochs = 100
    lambda_ae = 0.7
    lambda_kl = 0.01
    
    model = VAE(dim_mfc=23, pre_train=None)
    
    classifier_loss = list()
    ae_loss = list()
    kl_loss = list()
    
    for epoch in range(1,num_epochs+1):

        print('Epoch: %d' % epoch)

        start_time_epoch = time.time()
        n_samples = mfc_feats.shape[0]
        
        mfc_feats, labels = shuffle_feats_label(mfc_feats, labels)
        
        train_class_loss = list()
        train_ae_loss = list()
        train_kl_loss = list()

        for i in range(n_samples // mini_batch_size):

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            c_loss, a_loss, k_loss, embed, predict = model.train(mfc_features=mfc_feats[start:end], 
                                               labels=labels[start:end], 
                                               learning_rate=learning_rate, 
                                               lambda_ae=lambda_ae, 
                                               lambda_kl=lambda_kl)
            
            train_class_loss.append(c_loss)
            train_ae_loss.append(a_loss)
            train_kl_loss.append(k_loss)
        
        classifier_loss.append(np.mean(train_class_loss))
        ae_loss.append(np.mean(train_ae_loss))
        kl_loss.append(np.mean(train_kl_loss))
        print('Classifier Loss in epoch %d- %f' % (epoch, np.mean(train_class_loss)))
        print('AE Loss in epoch %d- %f' % (epoch, np.mean(train_ae_loss)))
        print('KLD Loss in epoch %d- %f' % (epoch, np.mean(train_kl_loss)))

        model.save(directory='./model', filename='VAE_net.ckpt')
        
        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, \
                (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))

        sys.stdout.flush()        




