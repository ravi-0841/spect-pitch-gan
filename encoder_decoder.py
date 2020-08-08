import tensorflow as tf
import numpy as np
import os
import scipy.io as scio
import time
import sys
import pylab


import modules.base_modules_default_init as basic_blocks

from utils.model_utils import l1_loss
from utils.feat_utils import shuffle_feats_label


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def encoder(input_mfc, reuse=False, scope_name='encoder'):
    
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

        o1 = basic_blocks.conv1d_layer(inputs=u2, filters=1, 
                kernel_size=15, strides=1, activation=None, name='embedding')

        classifier_branch = tf.layers.dense(inputs=o1, units=1, 
                activation=tf.nn.sigmoid, name='classifier')

        return tf.transpose(o1, [0,2,1]), tf.reduce_mean(tf.squeeze(classifier_branch, 
                           axis=-1), axis=-1, keep_dims=True, name='classifer_average')


def decoder(input_embed, reuse=False, final_filters=23, scope_name='decoder'):
    
    # expects the noise to be of shape [#batch, dim_noise, timesteps]
    input_embed_transposed = tf.transpose(input_embed, [0,2,1], 'input_embed_transpose')

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1_conv = basic_blocks.conv1d_layer(inputs=input_embed_transposed, 
                filters=64, kernel_size=15, strides=1, activation=None, 
                name='mfc_conv_1d')
        h1_gate = basic_blocks.conv1d_layer(inputs=input_embed_transposed, 
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


class AE(object):

    def __init__(self, encoder=encoder, decoder=decoder, dim_mfc=23, 
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

        # generate embedding and get reconstruction from AE
        self.embedding, self.prediction \
                = self.encoder(input_mfc=self.input_mfc, reuse=False, scope_name='encoder')

        self.reconstruction = self.decoder(input_embed=self.embedding, reuse=False, 
                scope_name='decoder')

        # Compute the AE and classification loss before reshaping
        self.ae_loss = l1_loss(y=self.input_mfc, y_hat=self.reconstruction)
        self.classification_loss = l1_loss(y=self.label, y_hat=self.prediction)

        # Compute full loss
        self.ae_class_loss = self.classification_loss + self.lambda_ae*self.ae_loss
        
        # get variables for optimization
        variables = tf.trainable_variables()
        self.trainable_vars = [var for var in variables if 'encoder' in var.name \
                                                        or 'decoder' in var.name]

        # generating embedding
        self.test_gen_embedding, self.test_prediction \
            = self.encoder(input_mfc=self.test_mfc, reuse=True)

        self.test_gen_mfc = self.decoder(input_embed=self.test_embedding, 
                                         reuse=True)


    def optimizer_initializer(self):
        
        self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                beta1=0.5).minimize(self.ae_class_loss, var_list=self.trainable_vars)


    def train(self, mfc_features, labels, lambda_ae, learning_rate):

        embeddings, class_loss, ae_loss, predictions, _ = self.sess.run([self.embedding, 
            self.classification_loss, self.ae_loss, self.prediction, self.optimizer], 
            feed_dict={self.input_mfc:mfc_features, 
                self.label:labels, self.lambda_ae:lambda_ae, 
                self.learning_rate:learning_rate})

        return class_loss, ae_loss, embeddings, predictions


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


    def load(self, filename):
        
        self.saver.restore(self.sess, filename)
    

    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)

        self.saver.save(self.sess, os.path.join(directory, filename))


#if __name__ == '__main__':
#
#    data = scio.loadmat('./data/neu-ang/train_mod_dtw_harvest.mat')
#    
#    mfc_A = np.vstack(np.transpose(data['src_mfc_feat'], (0,1,3,2)))
#    mfc_B = np.vstack(np.transpose(data['tar_mfc_feat'], (0,1,3,2)))
#
#    mfc_feats = np.concatenate((mfc_A, mfc_B), axis=0)    
#    labels = np.concatenate((np.zeros((mfc_A.shape[0],1)), 
#                             np.ones((mfc_B.shape[0],1))), axis=0)
#    
#    mini_batch_size = 512
#    learning_rate = 1e-03
#    num_epochs = 500
#    lambda_ae = 1.0
#    
#    model = AE(dim_mfc=23, pre_train=None)
#    
#    classifier_loss = list()
#    ae_loss = list()
#    
#    for epoch in range(1,num_epochs+1):
#
#        print('Epoch: %d' % epoch)
#
#        start_time_epoch = time.time()
#        n_samples = mfc_feats.shape[0]
#        
#        mfc_feats, labels = shuffle_feats_label(mfc_feats, labels)
#        
#        train_class_loss = list()
#        train_ae_loss = list()
#
#        for i in range(n_samples // mini_batch_size):
#
#            start = i * mini_batch_size
#            end = (i + 1) * mini_batch_size
#
#            c_loss, a_loss, embed, predict = model.train(mfc_features=mfc_feats[start:end], 
#                                               labels=labels[start:end], 
#                                               learning_rate=learning_rate, 
#                                               lambda_ae=lambda_ae)
#            
#            train_class_loss.append(c_loss)
#            train_ae_loss.append(a_loss)
#        
#        classifier_loss.append(np.mean(train_class_loss))
#        ae_loss.append(np.mean(train_ae_loss))
#        print('Classifier Loss in epoch %d- %f' % (epoch, np.mean(train_class_loss)))
#        print('AE Loss in epoch %d- %f' % (epoch, np.mean(train_ae_loss)))
#
#        model.save(directory='./model', filename='AE_net.ckpt')
#        
#        end_time_epoch = time.time()
#        time_elapsed_epoch = end_time_epoch - start_time_epoch
#
#        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, \
#                (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))
#
#        sys.stdout.flush()        


if __name__=='__main__':
    
    from mfcc_spect_analysis_VCGAN import _power_to_db
    import pyworld as pw
    import scipy.io.wavfile as scwav
    
    model = AE(dim_mfc=23)
    model.load('/home/ravi/Desktop/spect-pitch-gan/model/AE_net.ckpt')
    data = scio.loadmat('./data/neu-ang/valid_5.mat')
    mfc_A = data['src_mfc_feat']
    mfc_B = data['tar_mfc_feat']
    mfc_A = np.vstack(mfc_A)
    mfc_B = np.vstack(mfc_B)
    pre_A = model.get_prediction(mfc_features=np.transpose(mfc_A, [0,2,1]))
    pre_B = model.get_prediction(mfc_features=np.transpose(mfc_B, [0,2,1]))
    mfc_A = np.transpose(mfc_A, [0,2,1])
    mfc_B = np.transpose(mfc_B, [0,2,1])
    q = np.random.randint(0, mfc_A.shape[0] - 1)
    mfc_test = mfc_A[q:q+1]
    mfc_test_embed = model.get_embedding(mfc_features=mfc_test)
    mfc_test_recon = model.get_mfcc(embeddings=mfc_test_embed)
    
    mfc_rec = np.squeeze(mfc_test_recon)
    mfc_rec = np.copy(np.asarray(mfc_rec.T, np.float64), order='C')
    spect_rec = pw.decode_spectral_envelope(mfc_rec, 16000, 1024)
    mfc_test = np.squeeze(np.asarray(mfc_test, np.float64))
    mfc_test = np.copy(mfc_test.T, order='C')
    spect_test = pw.decode_spectral_envelope(mfc_test, 16000, 1024)
    pylab.subplot(121), pylab.imshow(np.squeeze(_power_to_db(spect_test.T ** 2)))
    pylab.title('Original')
    pylab.subplot(122), pylab.imshow(np.squeeze(_power_to_db(spect_rec.T ** 2)))
    pylab.title('Reconstructed')
    pylab.suptitle('Slice %d' % q)
    
    d = scwav.read('/home/ravi/Downloads/Emo-Conv/neutral-angry/all_above_0.5/angry/251.wav')
    d = np.asarray(d[1], np.float64)
    f0, sp, ap = pw.wav2world(d, 16000, frame_period=5)
    mfc = pw.code_spectral_envelope(sp, 16000, 23)
    embed = model.get_embedding(mfc_features=np.expand_dims(mfc.T, axis=0))
    mfc_recon = model.get_mfcc(embeddings=embed)
    mfc_recon = np.squeeze(mfc_recon)
    mfc_recon = np.copy(mfc_recon.T, order='C')
    spect_recon = pw.decode_spectral_envelope(np.asarray(mfc_recon, np.float64), 16000, 1024)
    speech_recon = pw.synthesize(f0, spect_recon[:len(f0)], ap, 16000, frame_period=5)
    speech_recon = (speech_recon - np.min(speech_recon)) / (np.max(speech_recon) - np.min(speech_recon))
    speech_recon = np.asarray(speech_recon - np.mean(speech_recon), np.float64)
    scwav.write('/home/ravi/Desktop/test_AE_5_l1.wav', 16000, speech_recon)














