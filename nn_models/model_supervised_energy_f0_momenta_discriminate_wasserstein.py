import os
import tensorflow as tf
import numpy as np

from modules.modules_supervised_energy_f0_momenta_discriminate_wasserstein import sampler_pitch, \
        sampler_energy, discriminator_pitch, discriminator_energy
import utils.model_utils as utils
from utils.tf_forward_tan import lddmm 

class VariationalCycleGAN(object):

    def __init__(self, dim_pitch=1, dim_energy=1, dim_mfc=23, 
            n_frames=128, discriminator_pitch=discriminator_pitch, 
            discriminator_energy=discriminator_energy, 
            sampler_pitch=sampler_pitch, sampler_energy=sampler_energy, 
            lddmm=lddmm, mode='train', log_file_name='no_name_passed', 
            pre_train=None):
        
        self.n_frames = n_frames
        self.pitch_shape = [None, dim_pitch, None] #[batch_size, num_features, num_frames]
        self.energy_shape = [None, dim_energy, None]
        self.mfc_shape = [None, dim_mfc, None]
            
        self.first_order_diff_mat = np.eye(self.n_frames, dtype=np.float32)
        for i in range(1, self.n_frames):
            self.first_order_diff_mat[i-1,i] = -1

        # Create the kernel for lddmm
        self.kernel_pitch = tf.expand_dims(tf.constant([6,50], 
            dtype=tf.float32), axis=0)
        self.kernel_energy = tf.expand_dims(tf.constant([6,5], 
            dtype=tf.float32), axis=0)

        self.sampler_pitch = sampler_pitch
        self.sampler_energy = sampler_energy
        self.discriminator_pitch = discriminator_pitch
        self.discriminator_energy = discriminator_energy
        self.lddmm_pitch = lddmm
        self.lddmm_energy = lddmm
        self.mode = mode

        self.build_model()
        self.optimizer_initializer()
        self.clip_discriminator_weights(0.1)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        if pre_train is not None:
            self.saver.restore(self.sess, pre_train)
        else:
            self.sess.run(tf.global_variables_initializer())

        if self.mode == 'train':
            self.train_step = 0
            self.writer = tf.summary.FileWriter('./tensorboard_log/'+log_file_name, 
                    tf.get_default_graph())
            self.generator_summaries, self.discriminator_summaries = self.summary()

    def build_model(self):

        # Placeholders for real training samples
        self.pitch_A_real = tf.placeholder(tf.float32, shape=self.pitch_shape, 
                name='pitch_A_real')
        self.pitch_B_real = tf.placeholder(tf.float32, shape=self.pitch_shape, 
                name='pitch_B_real')

        self.energy_A_real = tf.placeholder(tf.float32, shape=self.energy_shape, 
                name='energy_A_real')
        self.energy_B_real = tf.placeholder(tf.float32, shape=self.energy_shape, 
                name='energy_B_real')
        
        self.mfc_A = tf.placeholder(tf.float32, shape=self.mfc_shape, 
                name='mfc_A')
        self.mfc_B = tf.placeholder(tf.float32, shape=self.mfc_shape, 
                name='mfc_B')

        self.momenta_energy_A2B_real = tf.placeholder(tf.float32, shape=self.energy_shape, 
                name='momenta_energy_A2B_real')
        self.momenta_energy_B2A_real = tf.placeholder(tf.float32, shape=self.energy_shape, 
                name='momenta_energy_B2A_real')

        self.momenta_pitch_A2B_real = tf.placeholder(tf.float32, shape=self.pitch_shape, 
                name='momenta_pitch_A2B_real')
        self.momenta_pitch_B2A_real = tf.placeholder(tf.float32, shape=self.pitch_shape, 
                name='momenta_pitch_B2A_real')

        # Placeholders for fake generated samples
        self.pitch_A_fake = tf.placeholder(tf.float32, shape=self.pitch_shape, 
                name='pitch_A_fake')
        self.pitch_B_fake = tf.placeholder(tf.float32, shape=self.pitch_shape, 
                name='pitch_B_fake')

        self.energy_A_fake = tf.placeholder(tf.float32, shape=self.energy_shape, 
                name='energy_A_fake')
        self.energy_B_fake = tf.placeholder(tf.float32, shape=self.energy_shape, 
                name='energy_B_fake')

        # Placeholder for test samples
        self.pitch_A_test = tf.placeholder(tf.float32, shape=self.pitch_shape, 
                name='pitch_A_test')
        self.energy_A_test = tf.placeholder(tf.float32, shape=self.energy_shape, 
                name='energy_A_test')
        self.mfc_A_test = tf.placeholder(tf.float32, shape=self.mfc_shape, 
                name='mfc_A_test')

        self.pitch_B_test = tf.placeholder(tf.float32, shape=self.pitch_shape, 
                name='pitch_B_test')
        self.energy_B_test = tf.placeholder(tf.float32, shape=self.energy_shape, 
                name='energy_B_test')
        self.mfc_B_test = tf.placeholder(tf.float32, shape=self.mfc_shape, 
                name='mfc_B_test')

        # Place holder for lambda_cycle and lambda_identity
        self.lambda_energy = tf.placeholder(tf.float32, None, name='lambda_energy')
        self.lambda_momenta_energy = tf.placeholder(tf.float32, None, name='lambda_momenta_energy')
        self.lambda_momenta_pitch = tf.placeholder(tf.float32, None, name='lambda_momenta_pitch')

        '''
        Generator A
        '''
        # Generate pitch and energy from A to B
        self.momenta_pitch_A2B = self.sampler_pitch(input_pitch=self.pitch_A_real, 
                input_mfc=self.mfc_A, reuse=False, scope_name='sampler_pitch_A2B')
        self.pitch_A2B_fake = self.lddmm_pitch(x=self.pitch_A_real, p=self.momenta_pitch_A2B, 
                kernel=self.kernel_pitch, reuse=False, scope_name='lddmm_pitch')
        self.momenta_energy_A2B = self.sampler_energy(input_pitch=self.pitch_A2B_fake, 
                input_mfc=self.mfc_A, reuse=False, scope_name='sampler_energy_A2B')
        self.energy_A2B_fake = self.lddmm_energy(x=self.energy_A_real, p=self.momenta_energy_A2B, 
                kernel=self.kernel_energy, reuse=False, scope_name='lddmm_energy')


        '''
        Generator B
        '''
        # Generate pitch and energy from B to A
        self.momenta_pitch_B2A = self.sampler_pitch(input_pitch=self.pitch_B_real, 
                input_mfc=self.mfc_B, reuse=False, scope_name='sampler_pitch_B2A')
        self.pitch_B2A_fake = self.lddmm_pitch(x=self.pitch_B_real, p=self.momenta_pitch_B2A, 
                kernel=self.kernel_pitch, reuse=True, scope_name='lddmm_pitch')
        self.momenta_energy_B2A = self.sampler_energy(input_pitch=self.pitch_B2A_fake, 
                input_mfc=self.mfc_B, reuse=False, scope_name='sampler_energy_B2A')
        self.energy_B2A_fake = self.lddmm_energy(x=self.energy_B_real, p=self.momenta_energy_B2A, 
                kernel=self.kernel_energy, reuse=True, scope_name='lddmm_energy')

        '''
        Initialize the pitch discriminators
        '''
        # Discriminator initialized to keep parameters in memory
        self.pitch_discrimination_B_fake = self.discriminator_pitch(input_pitch=tf.concat([self.pitch_A_real, 
                self.pitch_A2B_fake], axis=1), reuse=False, scope_name='discriminator_pitch_A')

        self.pitch_discrimination_A_fake = self.discriminator_pitch(input_pitch=tf.concat([self.pitch_B_real, 
                self.pitch_B2A_fake], axis=1), reuse=False, scope_name='discriminator_pitch_B')

        '''
        Initialize the energy discriminators
        '''
        # Discriminator initialized to keep parameters in memory
        self.energy_discrimination_B_fake = self.discriminator_energy(input_energy=tf.concat([self.energy_A_real, 
                self.energy_A2B_fake], axis=1), reuse=False, scope_name='discriminator_energy_A')

        self.energy_discrimination_A_fake = self.discriminator_energy(input_energy=tf.concat([self.energy_B_real, 
                self.energy_B2A_fake], axis=1), reuse=False, scope_name='discriminator_energy_B')

        # Generator Loss
        self.generator_loss_momenta_pitch = (utils.l1_loss(self.momenta_pitch_A2B, self.momenta_pitch_A2B_real) \
                + utils.l1_loss(self.momenta_pitch_B2A, self.momenta_pitch_B2A_real)) / 2.0
        self.generator_loss_pitch = (utils.l1_loss(self.pitch_A2B_fake, self.pitch_B_real) \
                + utils.l1_loss(self.pitch_B2A_fake, self.pitch_A_real)) / 2.0
        self.generator_loss_momenta_energy = (utils.l1_loss(self.momenta_energy_A2B, self.momenta_energy_A2B_real) \
                + utils.l1_loss(self.momenta_energy_B2A, self.momenta_energy_B2A_real)) / 2.0
        self.generator_loss_energy = (utils.l1_loss(self.energy_A2B_fake, self.energy_B_real) \
                + utils.l1_loss(self.energy_B2A_fake, self.energy_A_real)) / 2.0
        self.generator_loss = self.generator_loss_pitch + self.lambda_energy*self.generator_loss_energy \
                + self.lambda_momenta_pitch*self.generator_loss_momenta_pitch \
                + self.lambda_momenta_energy*self.generator_loss_momenta_energy

        # Compute the pitch discriminator probability for pair of inputs
        self.pitch_discrimination_input_A_real_B_fake \
            = self.discriminator_pitch(input_pitch=tf.concat([self.pitch_A_real, self.pitch_B_fake], axis=1), 
                    reuse=True, scope_name='discriminator_pitch_A')
        self.pitch_discrimination_input_A_fake_B_real \
            = self.discriminator_pitch(input_pitch=tf.concat([self.pitch_A_fake, self.pitch_B_real], axis=1), 
                    reuse=True, scope_name='discriminator_pitch_A')

        self.pitch_discrimination_input_B_real_A_fake \
            = self.discriminator_pitch(input_pitch=tf.concat([self.pitch_B_real, self.pitch_A_fake], axis=1), 
                    reuse=True, scope_name='discriminator_pitch_B')
        self.pitch_discrimination_input_B_fake_A_real \
            = self.discriminator_pitch(input_pitch=tf.concat([self.pitch_B_fake, self.pitch_A_real], axis=1), 
                    reuse=True, scope_name='discriminator_pitch_B')
        
        # Compute pitch discriminator loss for backprop
        self.pitch_discriminator_loss_A \
            = (self.pitch_discrimination_input_A_real_B_fake \
                - self.pitch_discrimination_input_A_fake_B_real)
        self.pitch_discriminator_loss_B \
            = (self.pitch_discrimination_input_B_real_A_fake \
                - self.pitch_discrimination_input_B_fake_A_real)

        # Compute the energy discriminator probability for energy 
        self.energy_discrimination_input_A_real_B_fake \
            = self.discriminator_energy(input_energy=tf.concat([self.energy_A_real, self.energy_B_fake], axis=1), 
                    reuse=True, scope_name='discriminator_energy_A')
        self.energy_discrimination_input_A_fake_B_real \
            = self.discriminator_energy(input_energy=tf.concat([self.energy_A_fake, self.energy_B_real], axis=1), 
                    reuse=True, scope_name='discriminator_energy_A')

        self.energy_discrimination_input_B_real_A_fake \
            = self.discriminator_energy(input_energy=tf.concat([self.energy_B_real, self.energy_A_fake], axis=1), 
                    reuse=True, scope_name='discriminator_energy_B')
        self.energy_discrimination_input_B_fake_A_real \
            = self.discriminator_energy(input_energy=tf.concat([self.energy_B_fake, self.energy_A_real], axis=1), 
                    reuse=True, scope_name='discriminator_energy_B')
        
        # Compute pitch discriminator loss for backprop
        self.energy_discriminator_loss_A \
            = (self.energy_discrimination_input_A_real_B_fake \
                - self.energy_discrimination_input_A_fake_B_real)
        self.energy_discriminator_loss_B \
            = (self.energy_discrimination_input_B_real_A_fake \
                - self.energy_discrimination_input_B_fake_A_real)

        self.discriminator_A_loss = (self.pitch_discriminator_loss_A + self.energy_discriminator_loss_A)
        self.discriminator_B_loss = (self.pitch_discriminator_loss_B + self.energy_discriminator_loss_B)

        # Final merging of pitch and mfc discriminators
        self.discriminator_loss = (self.discriminator_A_loss + self.discriminator_B_loss) / 2.0

        # Categorize variables to optimize the two sets separately
        trainable_variables = tf.trainable_variables()
        self.discriminator_vars = [var for var in trainable_variables if 'discriminator' in var.name]
        self.generator_vars = [var for var in trainable_variables if 'sampler' in var.name]

    def optimizer_initializer(self):
        
        self.generator_learning_rate = tf.placeholder(tf.float32, None, 
                name='generator_learning_rate')
        self.discriminator_learning_rate = tf.placeholder(tf.float32, None, 
                name='discriminator_learning_rate')

        self.discriminator_train_op \
            = tf.train.AdamOptimizer(learning_rate=self.discriminator_learning_rate, \
                beta1=0.5).minimize(self.discriminator_loss, var_list=self.discriminator_vars)

        self.generator_train_op \
            = tf.train.AdamOptimizer(learning_rate=self.generator_learning_rate, \
                beta1=0.5).minimize(self.generator_loss, var_list=self.generator_vars)


    def clip_discriminator_weights(self, clip_range):

        self.clip_weights = [tf.assign(var, tf.clip_by_value(var, clip_value_min=-1*clip_range, 
            clip_value_max=clip_range)) for var in self.discriminator_vars]


    def train(self, pitch_A, mfc_A, energy_A, momenta_pitch_A2B, momenta_energy_A2B, 
            pitch_B, mfc_B, energy_B, momenta_pitch_B2A, momenta_energy_B2A, 
            lambda_energy, lambda_momenta_pitch, lambda_momenta_energy, 
            generator_learning_rate, discriminator_learning_rate):

        momenta_pitch_B, generation_pitch_B, momenta_energy_B, \
        generation_energy_B, momenta_pitch_A, generation_pitch_A, \
        momenta_energy_A, generation_energy_A, generator_loss, \
        generator_loss_pitch, generator_loss_energy, \
        _, generator_summaries = self.sess.run([self.momenta_pitch_A2B, self.pitch_A2B_fake, 
                    self.momenta_energy_A2B, self.energy_A2B_fake, self.momenta_pitch_B2A, 
                    self.pitch_B2A_fake, self.momenta_energy_B2A, self.energy_B2A_fake, 
                    self.generator_loss, self.generator_loss_pitch, self.generator_loss_energy, 
                    self.generator_train_op, self.generator_summaries], 
                    feed_dict = {self.lambda_energy:lambda_energy, 
                        self.pitch_A_real:pitch_A, 
                        self.pitch_B_real:pitch_B, self.mfc_A:mfc_A, 
                        self.mfc_B:mfc_B, self.energy_A_real:energy_A, 
                        self.energy_B_real:energy_B, 
                        self.momenta_pitch_A2B_real:momenta_pitch_A2B, 
                        self.momenta_energy_A2B_real:momenta_energy_A2B, 
                        self.momenta_pitch_B2A_real:momenta_pitch_B2A, 
                        self.momenta_energy_B2A_real:momenta_energy_B2A,
                        self.generator_learning_rate:generator_learning_rate})

        self.train_step += 1

        return generator_loss, generator_loss_pitch, generator_loss_energy, generation_pitch_A, \
                generation_energy_A, generation_pitch_B, generation_energy_B, \
                momenta_pitch_A, momenta_pitch_B, momenta_energy_A, momenta_energy_B


    def test_gen(self, mfc_A, pitch_A, energy_A, mfc_B, pitch_B, energy_B):
        gen_pitch_B, gen_energy_B, mom_pitch_B, mom_energy_B = self.sess.run([self.pitch_A2B_test, \
                                        self.energy_A2B_test, self.momenta_pitch_A2B_test, \
                                        self.momenta_energy_A2B_test], \
                                        feed_dict={self.pitch_A_test:pitch_A, \
                                        self.mfc_A_test:mfc_A, self.energy_A_test:energy_A})


        gen_pitch_A, gen_energy_A, mom_pitch_A, mom_energy_A = self.sess.run([self.pitch_B2A_test, \
                                        self.energy_B2A_test, self.momenta_pitch_B2A_test, \
                                        self.momenta_energy_B2A_test], \
                                        feed_dict={self.pitch_B_test:pitch_B, \
                                        self.mfc_B_test:mfc_B, self.energy_B_test:energy_B})
        
        return gen_pitch_A, gen_energy_A, gen_pitch_B, gen_energy_B, mom_pitch_A, mom_pitch_B, \
                mom_energy_A, mom_energy_B


    def test(self, input_pitch, input_energy, input_mfc, direction):

        if direction == 'A2B':
            generation_pitch, generation_energy, generation_mfc \
                    = self.sess.run([self.pitch_A2B_test, self.energy_A2B_test,  
                        self.mfc_A2B_test], feed_dict = {self.pitch_A_test:input_pitch, 
                        self.energy_A_test:input_energy, self.mfc_A_test:input_mfc})
        
        elif direction == 'B2A':
            generation_pitch, generation_energy, generation_mfc \
                    = self.sess.run([self.pitch_B2A_test, self.energy_B2A_test, 
                        self.mfc_B2A_test], feed_dict = {self.pitch_B_test:input_pitch, 
                        self.energy_B_test:input_energy, self.mfc_B_test:input_mfc})
        else:
            raise Exception('Conversion direction must be specified.')

        return generation_pitch, generation_energy, generation_mfc


    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, \
                        os.path.join(directory, filename))


    def load(self, filepath):

        self.saver.restore(self.sess, filepath)


    def summary(self):

        with tf.name_scope('generator_summaries'):
            generator_loss_summary = tf.summary.scalar('generator_loss', 
                    tf.reduce_mean(self.generator_loss))
            generator_summaries = tf.summary.merge([generator_loss_summary])

        with tf.name_scope('discriminator_summaries'):
            discriminator_loss_A_summary = tf.summary.scalar('discriminator_loss_A', 
                        tf.reduce_mean(self.discriminator_A_loss))
            discriminator_loss_B_summary = tf.summary.scalar('discriminator_loss_B', 
                    tf.reduce_mean(self.discriminator_B_loss))
            discriminator_loss_summary = tf.summary.scalar('discriminator_loss', 
                    tf.reduce_mean(self.discriminator_loss))
            discriminator_summaries = tf.summary.merge([discriminator_loss_A_summary, 
                discriminator_loss_B_summary, discriminator_loss_summary])

        return generator_summaries, discriminator_summaries


if __name__ == '__main__':
    
    model = VariationalCycleGAN(dim_pitch=1, dim_mfc=23)
    print('Graph Compile Successful.')
