import os
import tensorflow as tf
import numpy as np

from modules.modules_spect_mmd import sampler, \
        generator, joint_discriminator, spect_kernel
from utils.model_utils import l1_loss, eva_kernel
from utils.tf_forward_tan import lddmm 

class VariationalCycleGAN(object):

    def __init__(self, dim_pitch=1, dim_mfc=23, n_frames=128, 
            joint_discriminator=joint_discriminator, 
            spect_discriminator=spect_discriminator,  
            generator=generator, sampler=sampler, 
            lddmm=lddmm, mode='train', 
            log_dir='./log', pre_train=None):
        
        self.n_frames = n_frames
        self.pitch_shape = [None, dim_pitch, None] #[batch_size, num_features, num_frames]
        self.mfc_shape = [None, dim_mfc, None]

        self.first_order_diff_mat = np.eye(self.n_frames, dtype=np.float32)
        for i in range(1, self.n_frames):
            self.first_order_diff_mat[i-1,i] = -1

        # Create the kernel for lddmm
        self.kernel = tf.expand_dims(tf.constant([6,50], 
            dtype=tf.float32), axis=0)

        self.sampler = sampler
        self.generator = generator
        self.joint_discriminator = joint_discriminator
        self.spect_kernel = spect_kernel
        self.lddmm = lddmm
        self.mode = mode

        self.build_model()
        self.optimizer_initializer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        if pre_train is not None:
            self.saver.restore(self.sess, pre_train)
        else:
            self.sess.run(tf.global_variables_initializer())

        if self.mode == 'train':
            self.train_step = 0

    def build_model(self):

        # Placeholders for real training samples
        self.pitch_A_real = tf.placeholder(tf.float32, shape=self.pitch_shape, 
                name='pitch_A_real')
        self.pitch_B_real = tf.placeholder(tf.float32, shape=self.pitch_shape, 
                name='pitch_B_real')

        self.mfc_A_real = tf.placeholder(tf.float32, shape=self.mfc_shape, 
                name='mfc_A_real')
        self.mfc_B_real = tf.placeholder(tf.float32, shape=self.mfc_shape, 
                name='mfc_B_real')
        
        # Placeholders for fake generated samples
        self.pitch_A_fake = tf.placeholder(tf.float32, shape=self.pitch_shape, 
                name='pitch_A_fake')
        self.pitch_B_fake = tf.placeholder(tf.float32, shape=self.pitch_shape, 
                name='pitch_B_fake')

        self.mfc_A_fake = tf.placeholder(tf.float32, shape=self.mfc_shape, 
                name='mfc_A_fake')
        self.mfc_B_fake = tf.placeholder(tf.float32, shape=self.mfc_shape, 
                name='mfc_B_fake')

        # Placeholders for momenta variables
        self.momenta_A2B_real = tf.placeholder(tf.float32, shape=self.pitch_shape, 
                name='momenta_A2B_real')
        self.momenta_B2A_real = tf.placeholder(tf.float32, shape=self.pitch_shape, 
                name='momenta_B2A_real')

        # Placeholder for test samples
        self.pitch_A_test = tf.placeholder(tf.float32, shape=self.pitch_shape, 
                name='pitch_A_test')
        self.mfc_A_test = tf.placeholder(tf.float32, shape=self.mfc_shape, 
                name='mfc_A_test')

        self.pitch_B_test = tf.placeholder(tf.float32, shape=self.pitch_shape, 
                name='pitch_B_test')
        self.mfc_B_test = tf.placeholder(tf.float32, shape=self.mfc_shape, 
                name='mfc_B_test')

        # Place holder for lambda_cycle and lambda_identity
        self.lambda_pitch = tf.placeholder(tf.float32, None, 
                name='lambda_cycle_pitch')
        self.lambda_mfc = tf.placeholder(tf.float32, None, 
                name='lambda_cycle_mfc')
        self.lambda_momenta = tf.placeholder(tf.float32, None, 
                name='lambda_momenta')

        '''
        Generator A
        '''
        # Generate pitch from A to B
        self.momenta_generation_A2B = self.sampler(input_pitch=self.pitch_A_real, 
                input_mfc=self.mfc_A_real, reuse=False, scope_name='sampler_A2B')
        self.pitch_generation_A2B = self.lddmm(x=self.pitch_A_real, 
                p=self.momenta_generation_A2B, kernel=self.kernel, reuse=True, scope_name='lddmm')
        self.mfc_generation_A2B = self.generator(input_pitch=self.pitch_generation_A2B, 
                input_mfc=self.mfc_A_real, reuse=False, scope_name='generator_A2B')


        '''
        Generator B
        '''
        # Generate pitch from B to A
        self.momenta_generation_B2A = self.sampler(input_pitch=self.pitch_B_real, 
                input_mfc=self.mfc_B_real, reuse=False, scope_name='sampler_B2A')
        self.pitch_generation_B2A = self.lddmm(x=self.pitch_B_real, 
                p=self.momenta_generation_B2A, kernel=self.kernel, reuse=True, scope_name='lddmm')
        self.mfc_generation_B2A = self.generator(input_pitch=self.pitch_generation_B2A, 
                input_mfc=self.mfc_B_real, reuse=False, scope_name='generator_B2A')

        '''
        Initialize the joint discriminators
        '''
        # Discriminator initialized to keep parameters in memory
        self.joint_discrimination_B_fake = self.joint_discriminator(input_mfc=tf.concat([self.mfc_A_real, 
            self.mfc_generation_A2B], axis=1), input_pitch=tf.concat([self.pitch_A_real, 
                self.pitch_generation_A2B], axis=1), reuse=False, scope_name='joint_discriminator_A')

        self.joint_discrimination_A_fake = self.joint_discriminator(input_mfc=tf.concat([self.mfc_B_real, 
            self.mfc_generation_B2A], axis=1), input_pitch=tf.concat([self.pitch_B_real, 
                self.pitch_generation_B2A], axis=1), reuse=False, scope_name='joint_discriminator_B')

        '''
        Initialize the spect kernel
        '''
        # Kernel initialized to keep parameters in memory
        self.spect_kernel_A_real = self.spect_kernel(input_mfc=self.mfc_A_real, 
                reuse=False, scope_name='spect_kernel')
        self.spect_kernel_A_fake = self.spect_kernel(input_mfc=self.mfc_generation_B2A, 
                reuse=True, scope_name='spect_kernel')
        self.spect_kernel_B_real = self.spect_kernel(input_mfc=self.mfc_B_real, 
                reuse=True, scope_name='spect_kernel')
        self.spect_kernel_B_fake = self.spect_kernel(input_mfc=self.mfc_generation_A2B, 
                reuse=True, scope_name='spect_kernel')

        self.kernel_AA = eval_kernel(kernel1=self.spect_kernel_A_real, 
                kernel2=self.spect_kernel_A_fake)
        self.kernel_BB = eval_kernel(kernel1=self.spect_kernel_B_real, 
                kernel2=self.spect_kernel_B_fake)
        self.kernel_AB = eval_kernel(kernel1=self.spect_kernel_A_real, 
                kernel2=self.spect_kernel_B_real)


        '''
        Computing loss for generators
        '''
        self.momenta_loss_A2B = l1_loss(self.momenta_A2B_real, self.momenta_generation_A2B)
        self.momenta_loss_B2A = l1_loss(self.momenta_B2A_real, self.momenta_generation_B2A)

        self.pitch_loss_A2B = l1_loss(self.pitch_B_real, self.pitch_generation_A2B)
        self.pitch_loss_B2A = l1_loss(self.pitch_A_real, self.pitch_generation_B2A)

        self.mfc_loss_A2B = l1_loss(self.mfc_B_real, self.mfc_generation_A2B)
        self.mfc_loss_B2A = l1_loss(self.mfc_A_real, self.mfc_generation_B2A)

        # Merge the loss for generators A2B
        self.loss_A2B \
            = self.lambda_pitch*self.pitch_loss_A2B + self.lambda_mfc*self.mfc_loss_A2B \
                + self.lambda_momenta * self.momenta_loss_A2B

        self.loss_B2A \
            = self.lambda_pitch*self.pitch_loss_B2A + self.lambda_mfc*self.mfc_loss_B2A \
                + self.lambda_momenta * self.momenta_loss_B2A

        self.generator_loss = self.loss_A2B + self.loss_B2A 
        
        # Compute the discriminator probability for pair of inputs
        self.joint_discrimination_input_A_real_B_fake \
            = self.joint_discriminator(input_mfc=tf.concat([self.mfc_A_real, self.mfc_B_fake], axis=1), 
                    input_pitch=tf.concat([self.pitch_A_real, self.pitch_B_fake], axis=1), 
                    reuse=True, scope_name='joint_discriminator_A')
        self.joint_discrimination_input_A_fake_B_real \
            = self.joint_discriminator(input_mfc=tf.concat([self.mfc_A_fake, self.mfc_B_real], axis=1), 
                    input_pitch=tf.concat([self.pitch_A_fake, self.pitch_B_real], axis=1), 
                    reuse=True, scope_name='joint_discriminator_A')

        self.joint_discrimination_input_B_real_A_fake \
            = self.joint_discriminator(input_mfc=tf.concat([self.mfc_B_real, self.mfc_A_fake], axis=1), 
                    input_pitch=tf.concat([self.pitch_B_real, self.pitch_A_fake], axis=1), 
                    reuse=True, scope_name='joint_discriminator_B')
        self.joint_discrimination_input_B_fake_A_real \
            = self.joint_discriminator(input_mfc=tf.concat([self.mfc_B_fake, self.mfc_A_real], axis=1), 
                    input_pitch=tf.concat([self.pitch_B_fake, self.pitch_A_real], axis=1), 
                    reuse=True, scope_name='joint_discriminator_B')
        
        # Compute discriminator loss for backprop
        self.joint_discriminator_loss_input_A_real \
            = l1_loss(y=tf.zeros_like(self.joint_discrimination_input_A_real_B_fake), 
                    y_hat=self.joint_discrimination_input_A_real_B_fake)
        self.joint_discriminator_loss_input_A_fake \
            = l1_loss(y=tf.ones_like(self.joint_discrimination_input_A_fake_B_real), 
                    y_hat=self.joint_discrimination_input_A_fake_B_real)
        self.joint_discriminator_loss_A = (self.joint_discriminator_loss_input_A_real \
                                     + self.joint_discriminator_loss_input_A_fake) / 2.0

        self.joint_discriminator_loss_input_B_real \
            = l1_loss(y=tf.zeros_like(self.joint_discrimination_input_B_real_A_fake), 
                    y_hat=self.joint_discrimination_input_B_real_A_fake)
        self.joint_discriminator_loss_input_B_fake \
            = l1_loss(y=tf.ones_like(self.joint_discrimination_input_B_fake_A_real), 
                    y_hat=self.joint_discrimination_input_B_fake_A_real)
        self.joint_discriminator_loss_B = (self.joint_discriminator_loss_input_B_real \
                                     + self.joint_discriminator_loss_input_B_fake) / 2.0

        # Merge the two discriminators into one
        self.joint_discriminator_loss = (self.joint_discriminator_loss_A + self.joint_discriminator_loss_B) / 2.0


        # Evaluate the kernel for mfcc 
        self.spect_kernel_disc_A_real = self.spect_kernel(input_mfc=self.mfc_A_real, 
                reuse=True, scope_name='spect_kernel')
        self.spect_kernel_disc_A_fake = self.spect_kernel(input_mfc=self.mfc_A_fake, 
                reuse=True, scope_name='spect_kernel')
        self.spect_kernel_disc_B_real = self.spect_kernel(input_mfc=self.mfc_B_real, 
                reuse=True, scope_name='spect_kernel')
        self.spect_kernel_disc_B_fake = self.spect_kernel(input_mfc=self.mfc_B_fake, 
                reuse=True, scope_name='spect_kernel')

        self.spect_kernel_A_real_B_real = eval_kernel(kernel1=self.spect_kernel_disc_A_real, 
                kernel2=self.spect_kernel_disc_B_real)
        self.spect_kernel_A_real_A_fake = eval_kernel(kernel1=self.spect_kernel_disc_A_real, 
                kernel2=self.spect_kernel_disc_A_fake)
        self.spect_kernel_B_real_B_fake = eval_kernel(kernel1=self.spect_kernel_disc_B_real, 
                kernel2=self.spect_kernel_disc_B_fake)

        # Merge the two discriminators into one
        self.spect_kernel_loss = 2*self.spect_kernel_A_real_B_real \
                                - self.spect_kernel_A_real_A_fake \
                                - self.spect_kernel_B_real_B_fake 

        # Final merging of joint and spect discriminators
        self.discriminator_loss = self.joint_discriminator_loss + self.spect_kernel_loss

        # Categorize variables to optimize the two sets separately
        trainable_variables = tf.trainable_variables()
        self.discriminator_vars = [var for var in trainable_variables if 'discriminator' in var.name \
                                                                    or 'kernel' in var.name]
        self.generator_vars = [var for var in trainable_variables if 'generator' in var.name \
                                                                    or 'sampler' in var.name]

        # Reserved for test
        self.momenta_A2B_test = self.sampler(input_pitch=self.pitch_A_test, 
                input_mfc=self.mfc_A_test, reuse=True, scope_name='sampler_A2B')
        self.pitch_A2B_test = self.lddmm(x=self.pitch_A_test, 
                p=self.momenta_A2B_test, kernel=self.kernel, reuse=True, scope_name='lddmm')
        self.mfc_A2B_test = self.generator(input_pitch=self.pitch_A2B_test, 
                input_mfc=self.mfc_A_test, reuse=True, scope_name='generator_A2B')

        self.momenta_B2A_test = self.sampler(input_pitch=self.pitch_B_test, 
                input_mfc=self.mfc_B_test, reuse=True, scope_name='sampler_B2A')
        self.pitch_B2A_test = self.lddmm(x=self.pitch_B_test, 
                p=self.momenta_B2A_test, kernel=self.kernel, reuse=True, scope_name='lddmm')
        self.mfc_B2A_test = self.generator(input_pitch=self.pitch_B2A_test, 
                input_mfc=self.mfc_B_test, reuse=True, scope_name='generator_B2A')


    def optimizer_initializer(self):
        
        self.generator_learning_rate = tf.placeholder(tf.float32, None, 
                name='generator_learning_rate')
        self.discriminator_learning_rate = tf.placeholder(tf.float32, None, 
                name='discriminator_learning_rate')

        self.discriminator_optimizer \
                = tf.train.AdamOptimizer(learning_rate=self.discriminator_learning_rate, 
                        beta1=0.5).minimize(self.discriminator_loss, var_list=self.discriminator_vars)

        self.generator_optimizer \
                = tf.train.AdamOptimizer(learning_rate=self.generator_learning_rate, 
                        beta1=0.5).minimize(self.generator_loss, var_list=self.generator_vars)


    def train(self, pitch_A, mfc_A, momenta_A2B, pitch_B, mfc_B, 
            momenta_B2A, lambda_pitch, lambda_mfc, lambda_momenta, 
            generator_learning_rate):

        generated_momenta_B, generated_pitch_B, generated_mfc_B, \
                momenta_loss_A2B, pitch_loss_A2B, mfc_loss_A2B, generated_momenta_A, \
                generated_pitch_A, generated_mfc_A, momenta_loss_B2A, pitch_loss_B2A, \
                mfc_loss_B2A, _ \
                = self.sess.run([self.momenta_generation_A2B, self.pitch_generation_A2B, 
                    self.mfc_generation_A2B, self.momenta_loss_A2B, self.pitch_loss_A2B, 
                    self.mfc_loss_A2B, self.momenta_generation_B2A, self.pitch_generation_B2A, 
                    self.mfc_generation_B2A, self.momenta_loss_B2A, self.pitch_loss_B2A, 
                    self.mfc_loss_B2A, self.generator_optimizer], 
                    feed_dict = {self.lambda_pitch:lambda_pitch, 
                        self.lambda_mfc:lambda_mfc, self.lambda_momenta:lambda_momenta, 
                        self.pitch_A_real:pitch_A, self.mfc_A_real:mfc_A, 
                        self.momenta_A2B_real:momenta_A2B, 
                        self.momenta_B2A_real:momenta_B2A, 
                        self.pitch_B_real:pitch_B, self.mfc_B_real:mfc_B, 
                        self.generator_learning_rate:generator_learning_rate})

#        self.writer.add_summary(generator_summaries, self.train_step)
#        self.writer.add_summary(discriminator_summaries, self.train_step)

        self.train_step += 1

        return momenta_loss_A2B, momenta_loss_B2A, pitch_loss_A2B, pitch_loss_B2A, \
                mfc_loss_A2B, mfc_loss_B2A, generated_momenta_A, generated_pitch_A, \
                generated_mfc_A, generated_momenta_B, generated_pitch_B, generated_mfc_B


    def test(self, mfc_A, pitch_A, mfc_B, pitch_B):
        gen_mom_B, gen_pitch_B, gen_mfc_B, = self.sess.run([self.momenta_A2B_test, \
                                    self.pitch_A2B_test, self.mfc_A2B_test], \
                                    feed_dict={self.pitch_A_test:pitch_A, \
                                        self.mfc_A_test:mfc_A})


        gen_mom_A, gen_pitch_A, gen_mfc_A = self.sess.run([self.momenta_B2A_test, \
                                    self.pitch_B2A_test, self.mfc_B2A_test], \
                                    feed_dict={self.pitch_B_test:pitch_B, \
                                        self.mfc_B_test:mfc_B})
        
        return gen_pitch_A, gen_mfc_A, gen_pitch_B, gen_mfc_B, gen_mom_A, gen_mom_B

    def test_compact(self, input_pitch, input_mfc, direction):

        if direction == 'A2B':
            generated_pitch, generated_mfc = self.sess.run([self.pitch_A2B_test, 
                self.mfc_A2B_test], feed_dict = {self.pitch_A_test:input_pitch, 
                    self.mfc_A_test:input_mfc})
        
        elif direction == 'B2A':
            generated_pitch, generated_mfc = self.sess.run([self.pitch_B2A_test, 
                self.mfc_B2A_test], feed_dict = {self.pitch_B_test:input_pitch, 
                    self.mfc_B_test:input_mfc})
        else:
            raise Exception('Conversion direction must be specified.')

        return generated_pitch, generated_mfc


    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, \
                        os.path.join(directory, filename))

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)


    def summary(self):

        with tf.name_scope('generator_summaries'):
            cycle_loss_summary = tf.summary.scalar('cycle_loss', \
                                    self.cycle_loss)
            identity_loss_summary = tf.summary.scalar('identity_loss', \
                                    self.identity_loss)
            generator_loss_A2B_summary = tf.summary.scalar('generator_loss_A2B', \
                                    self.generator_loss_A2B)
            generator_loss_B2A_summary = tf.summary.scalar('generator_loss_B2A', \
                                    self.generator_loss_B2A)
            generator_loss_summary = tf.summary.scalar('generator_loss', \
                                    self.generator_loss)
            generator_summaries = tf.summary.merge([cycle_loss_summary, \
                                    identity_loss_summary, \
                                    generator_loss_A2B_summary, \
                                    generator_loss_B2A_summary, \
                                    generator_loss_summary])

        with tf.name_scope('discriminator_summaries'):
            discriminator_loss_A_summary \
                = tf.summary.scalar('discriminator_loss_A', \
                        self.discriminator_loss_A)
            discriminator_loss_B_summary \
                = tf.summary.scalar('discriminator_loss_B', \
                        self.discriminator_loss_B)
            discriminator_loss_summary \
                = tf.summary.scalar('discriminator_loss', \
                        self.discriminator_loss)
            discriminator_summaries \
                = tf.summary.merge([discriminator_loss_A_summary, \
                        discriminator_loss_B_summary, \
                        discriminator_loss_summary])

        return generator_summaries, discriminator_summaries


if __name__ == '__main__':
    
    model = VariationalCycleGAN(num_features = 23)
    print('Graph Compile Successful.')
