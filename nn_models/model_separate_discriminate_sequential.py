import os
import tensorflow as tf
import numpy as np

from modules.modules_separate_discriminate_sequential import generator, predictor, discriminator
from utils.model_utils import l1_loss
from utils.tf_forward_tan import forward_tan

class VariationalCycleGAN(object):

    def __init__(self, dim_pitch=1, dim_mfc=23, n_frames=128, 
            discriminator=discriminator, generator=generator,
            predictor=predictor, mode='train', log_dir_name='no_name_seq', 
            pre_train=None):
        
        self.n_frames = n_frames
        self.pitch_shape = [None, dim_pitch, None] #[batch_size, num_features, num_frames]
        self.mfc_shape = [None, dim_mfc, None]

        self.center_diff_mat = np.zeros((n_frames, n_frames), np.float32)
        for i in range(self.n_frames-1):
            self.center_diff_mat[i,i+1] = 0.5
        for i in range(1, self.n_frames):
            self.center_diff_mat[i,i-1] = -0.5
            
        self.first_order_diff_mat = np.eye(self.n_frames, dtype=np.float32)
        for i in range(1, self.n_frames):
            self.first_order_diff_mat[i-1,i] = -1

        # Create the kernel for lddmm
        self.kernel = tf.expand_dims(tf.constant([6,50], 
            dtype=tf.float32), axis=0)

        self.predictor = predictor
        self.generator = generator
        self.discriminator = discriminator
        self.mode = mode

        self.build_model()
        self.optimizer_initializer()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if pre_train is not None:
            vars_to_restore = [var for var in tf.global_variables() \
                                    if 'generator' in var.name]
            self.saver = tf.train.Saver(vars_to_restore)
            self.saver.restore(self.sess, pre_train)

        if self.mode == 'train':
            self.train_step = 0
            self.writer = tf.summary.FileWriter('./tensorboard_log/'+log_dir_name, 
                    tf.get_default_graph())
            self.predictor_summaries, self.discriminator_summaries = self.summary()

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
        self.lambda_cycle_mfc = tf.placeholder(tf.float32, None, 
                name='lambda_cycle_mfc')

        '''
        Generator A
        '''
        # Generate pitch from A to B
        self.momentum_A2B = self.generator(input_pitch=self.pitch_A_real, 
                input_mfc=self.mfc_A_real, reuse=False, scope_name='generator_A2B')
        self.pitch_generation_A2B = forward_tan(x=self.pitch_A_real, 
                p=self.momentum_A2B, kernel=self.kernel)
        self.mfc_generation_A2B = self.predictor(input_pitch=self.pitch_generation_A2B, 
                input_mfc=self.mfc_A_real, reuse=False, scope_name='predictor_A2B')
        # Cyclic generation
        self.momentum_cycle_A2A = self.generator(input_pitch=self.pitch_generation_A2B, 
                input_mfc=self.mfc_generation_A2B, reuse=False, scope_name='generator_B2A')
        self.pitch_cycle_A2A = forward_tan(x=self.pitch_generation_A2B, 
                p=self.momentum_cycle_A2A, kernel=self.kernel)
        self.mfc_cycle_A2A = self.predictor(input_pitch=self.pitch_cycle_A2A, 
                input_mfc=self.mfc_generation_A2B, reuse=False, scope_name='predictor_B2A')


        '''
        Generator B
        '''
        # Generate pitch from B to A
        self.momentum_B2A = self.generator(input_pitch=self.pitch_B_real, 
                input_mfc=self.mfc_B_real, reuse=True, scope_name='generator_B2A')
        self.pitch_generation_B2A = forward_tan(x=self.pitch_B_real, 
                p=self.momentum_B2A, kernel=self.kernel)
        self.mfc_generation_B2A = self.predictor(input_pitch=self.pitch_generation_B2A, 
                input_mfc=self.mfc_B_real, reuse=True, scope_name='predictor_B2A')
        # Cyclic generation
        self.momentum_cycle_B2B = self.generator(input_pitch=self.pitch_generation_B2A, 
                input_mfc=self.mfc_generation_B2A, reuse=True, scope_name='generator_A2B')
        self.pitch_cycle_B2B = forward_tan(x=self.pitch_generation_B2A, 
                p=self.momentum_cycle_B2B, kernel=self.kernel)
        self.mfc_cycle_B2B = self.predictor(input_pitch=self.pitch_cycle_B2B, 
                input_mfc=self.mfc_generation_B2A, reuse=True, scope_name='predictor_A2B')


        # Generator Discriminator Loss
        self.discrimination_B_fake = self.discriminator(input_mfc_1=self.mfc_A_real, 
            input_mfc_2=self.mfc_generation_A2B, reuse=False, scope_name='discriminator_A')

        self.discrimination_A_fake = self.discriminator(input_mfc_1=self.mfc_B_real, 
            input_mfc_2=self.mfc_generation_B2A, reuse=False, scope_name='discriminator_B')

        # Cycle loss
        self.cycle_loss_mfc = (l1_loss(y=self.mfc_A_real, 
            y_hat=self.mfc_cycle_A2A) + l1_loss(y=self.mfc_B_real, 
                y_hat=self.mfc_cycle_B2B)) / 2.0


        # Predictor loss
        # Predictor wants to fool discriminator
        self.predictor_loss_A2B = l1_loss(y=tf.ones_like(self.discrimination_B_fake), 
                y_hat=self.discrimination_B_fake)
        self.predictor_loss_B2A = l1_loss(y=tf.ones_like(self.discrimination_A_fake), 
                y_hat=self.discrimination_A_fake)
        self.pred_disc_loss = (self.predictor_loss_A2B + self.predictor_loss_B2A) / 2.0

        self.predictor_loss = self.pred_disc_loss + self.lambda_cycle_mfc*self.cycle_loss_mfc

        # Compute the discriminator probability for pair of inputs
        self.discrimination_input_A_real_B_fake \
            = self.discriminator(input_mfc_1=self.mfc_A_real, input_mfc_2=self.mfc_B_fake, 
                    reuse=True, scope_name='discriminator_A')
        self.discrimination_input_A_fake_B_real \
            = self.discriminator(input_mfc_1=self.mfc_A_fake, input_mfc_2=self.mfc_B_real, 
                    reuse=True, scope_name='discriminator_A')

        self.discrimination_input_B_real_A_fake \
            = self.discriminator(input_mfc_1=self.mfc_B_real, input_mfc_2=self.mfc_A_fake, 
                    reuse=True, scope_name='discriminator_B')
        self.discrimination_input_B_fake_A_real \
            = self.discriminator(input_mfc_1=self.mfc_B_fake, input_mfc_2=self.mfc_A_real, 
                    reuse=True, scope_name='discriminator_B')


        # Compute discriminator loss for backprop
        self.discriminator_loss_input_A_real \
            = l1_loss(y=tf.zeros_like(self.discrimination_input_A_real_B_fake), 
                    y_hat=self.discrimination_input_A_real_B_fake)
        self.discriminator_loss_input_A_fake \
            = l1_loss(y=tf.ones_like(self.discrimination_input_A_fake_B_real), 
                    y_hat=self.discrimination_input_A_fake_B_real)
        self.discriminator_loss_A = (self.discriminator_loss_input_A_real \
                                     + self.discriminator_loss_input_A_fake) / 2.0

        self.discriminator_loss_input_B_real \
            = l1_loss(y=tf.zeros_like(self.discrimination_input_B_real_A_fake), 
                    y_hat=self.discrimination_input_B_real_A_fake)
        self.discriminator_loss_input_B_fake \
            = l1_loss(y=tf.ones_like(self.discrimination_input_B_fake_A_real), 
                    y_hat=self.discrimination_input_B_fake_A_real)
        self.discriminator_loss_B = (self.discriminator_loss_input_B_real \
                                     + self.discriminator_loss_input_B_fake) / 2.0

        # Merge the two discriminators into one
        self.discriminator_loss = (self.discriminator_loss_A + self.discriminator_loss_B) / 2.0

        # Categorize variables to optimize the two sets separately
        trainable_variables = tf.trainable_variables()
        self.discriminator_vars = [var for var in trainable_variables if 'discriminator' in var.name]
        self.predictor_vars = [var for var in trainable_variables if 'predictor' in var.name]

        # Reserved for test
        self.momentum_A2B_test = self.generator(input_pitch=self.pitch_A_test, 
                input_mfc=self.mfc_A_test, reuse=True, scope_name='generator_A2B')
        self.pitch_A2B_test = forward_tan(x=self.pitch_A_test, 
                p=self.momentum_A2B_test, kernel=self.kernel)
        self.mfc_A2B_test = self.predictor(input_pitch=self.pitch_A2B_test, 
                input_mfc=self.mfc_A_test, reuse=True, scope_name='predictor_A2B')

        self.momentum_B2A_test = self.generator(input_pitch=self.pitch_B_test, 
                input_mfc=self.mfc_B_test, reuse=True, scope_name='generator_B2A')
        self.pitch_B2A_test = forward_tan(x=self.pitch_B_test, 
                p=self.momentum_B2A_test, kernel=self.kernel)
        self.mfc_B2A_test = self.predictor(input_pitch=self.pitch_B2A_test, 
                input_mfc=self.mfc_B_test, reuse=True, scope_name='predictor_B2A')


    def optimizer_initializer(self):
        
        self.predictor_learning_rate = tf.placeholder(tf.float32, None, 
                name='predictor_learning_rate')
        self.discriminator_learning_rate = tf.placeholder(tf.float32, None, 
                name='discriminator_learning_rate')
        self.discriminator_optimizer \
            = tf.train.AdamOptimizer(learning_rate=self.discriminator_learning_rate, \
                beta1=0.5)
        self.discriminator_grads \
                = self.discriminator_optimizer.compute_gradients(self.discriminator_loss, 
                        var_list=self.discriminator_vars)
        self.discriminator_train_op \
                = self.discriminator_optimizer.apply_gradients(self.discriminator_grads)

        self.predictor_optimizer \
            = tf.train.AdamOptimizer(learning_rate=self.predictor_learning_rate, \
                beta1=0.5)
        self.predictor_grads \
                = self.predictor_optimizer.compute_gradients(self.predictor_loss, 
                        var_list=self.predictor_vars)
        self.predictor_train_op \
                = self.predictor_optimizer.apply_gradients(self.predictor_grads)
                

    def train_grad(self, pitch_A, mfc_A, pitch_B, mfc_B, lambda_cycle_mfc, 
            predictor_learning_rate, discriminator_learning_rate):

        momentum_B, generation_pitch_B, generation_mfc_B, momentum_A, \
                generation_pitch_A, generation_mfc_A, predictor_loss, \
                predictor_grads, _, predictor_summaries \
                = self.sess.run([self.momentum_A2B, self.pitch_generation_A2B, 
                    self.mfc_generation_A2B, self.momentum_B2A, self.pitch_generation_B2A, 
                    self.mfc_generation_B2A, self.pred_disc_loss, self.predictor_grads, 
                    self.predictor_train_op, self.predictor_summaries], 
                    feed_dict = {self.lambda_cycle_mfc:lambda_cycle_mfc, 
                        self.pitch_A_real:pitch_A, self.pitch_B_real:pitch_B, 
                        self.mfc_A_real:mfc_A, self.mfc_B_real:mfc_B, 
                        self.predictor_learning_rate:predictor_learning_rate})

        self.writer.add_summary(predictor_summaries, self.train_step)

        discriminator_loss, discriminator_grads, _, discriminator_summaries \
            = self.sess.run([self.discriminator_loss, self.discriminator_grads, 
                self.discriminator_train_op, self.discriminator_summaries], 
                    feed_dict = {self.pitch_A_real:pitch_A, self.pitch_B_real:pitch_B, 
                        self.mfc_A_real:mfc_A, self.mfc_B_real:mfc_B, 
                        self.discriminator_learning_rate:discriminator_learning_rate, 
                        self.pitch_A_fake:generation_pitch_A, self.pitch_B_fake:generation_pitch_B, 
                        self.mfc_A_fake:generation_mfc_A, self.mfc_B_fake:generation_mfc_B})

        self.writer.add_summary(discriminator_summaries, self.train_step)

        self.train_step += 1

        return predictor_loss, discriminator_loss, generation_pitch_A, \
                generation_mfc_A, generation_pitch_B, generation_mfc_B, \
                momentum_A, momentum_B, predictor_grads, discriminator_grads


    def compute_gradients(self, pitch_A, mfc_A, pitch_B, mfc_B, 
                                lambda_cycle_pitch, lambda_cycle_mfc, 
                                lambda_momenta):
        
        predictor_gradient, discriminator_gradient \
            = self.sess.run([self.predictor_grads, self.discriminator_grads], 
                            feed_dict={self.lambda_cycle_pitch:lambda_cycle_pitch, 
                        self.lambda_cycle_mfc:lambda_cycle_mfc, 
                        self.lambda_momenta:lambda_momenta, self.pitch_A_real:pitch_A, 
                        self.pitch_B_real:pitch_B, self.mfc_A_real:mfc_A, 
                        self.mfc_B_real:mfc_B})
        return predictor_gradient, discriminator_gradient
    
    
    def test_gen(self, mfc_A, pitch_A, mfc_B, pitch_B):
        gen_mom_B, gen_pitch_B, gen_mfc_B, = self.sess.run([self.momentum_A2B_test, \
                                    self.pitch_A2B_test, self.mfc_A2B_test], \
                                    feed_dict={self.pitch_A_test:pitch_A, \
                                        self.mfc_A_test:mfc_A})


        gen_mom_A, gen_pitch_A, gen_mfc_A = self.sess.run([self.momentum_B2A_test, \
                                    self.pitch_B2A_test, self.mfc_B2A_test], \
                                    feed_dict={self.pitch_B_test:pitch_B, \
                                        self.mfc_B_test:mfc_B})
        
        return gen_mom_A, gen_pitch_A, gen_mfc_A, gen_mom_B, gen_pitch_B, gen_mfc_B

    def test(self, input_pitch, input_mfc, direction):

        if direction == 'A2B':
            generation_pitch, generation_mfc = self.sess.run([self.pitch_A2B_test, 
                self.mfc_A2B_test], feed_dict = {self.pitch_A_test:input_pitch, 
                    self.mfc_A_test:input_mfc})
        
        elif direction == 'B2A':
            generation_pitch, generation_mfc = self.sess.run([self.pitch_B2A_test, 
                self.mfc_B2A_test], feed_dict = {self.pitch_B_test:input_pitch, 
                    self.mfc_B_test:input_mfc})
        else:
            raise Exception('Conversion direction must be specified.')

        return generation_pitch, generation_mfc


    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, \
                        os.path.join(directory, filename))
        
#        return os.path.join(directory, filename)

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)


    def summary(self):

        with tf.name_scope('predictor_summaries'):
            cycle_loss_mfc_summary = tf.summary.scalar('cycle_loss_mfc', 
                    self.cycle_loss_mfc)
            predictor_loss_A2B_summary = tf.summary.scalar('predictor_loss_A2B', 
                    self.predictor_loss_A2B)
            predictor_loss_B2A_summary = tf.summary.scalar('predictor_loss_B2A', 
                    self.predictor_loss_B2A)
            predictor_loss_summary = tf.summary.scalar('predictor_loss', 
                    self.pred_disc_loss)
            predictor_gradient_summary = tf.summary.scalar('predictor_gradients', 
                    tf.divide(tf.reduce_sum(self.predictor_grads[0][0]**2), 
                    tf.reduce_sum(self.predictor_grads[0][1]**2)))
            predictor_summaries = tf.summary.merge([cycle_loss_mfc_summary, 
                predictor_loss_A2B_summary, predictor_loss_B2A_summary, 
                predictor_loss_summary, predictor_gradient_summary])

        with tf.name_scope('discriminator_summaries'):
            discriminator_loss_A_summary = tf.summary.scalar('discriminator_loss_A', 
                        self.discriminator_loss_A)
            discriminator_loss_B_summary = tf.summary.scalar('discriminator_loss_B', 
                    self.discriminator_loss_B)
            discriminator_loss_summary = tf.summary.scalar('discriminator_loss', 
                    self.discriminator_loss)
            discriminator_gradient_summary = tf.summary.scalar('discriminator_gradients', 
                    tf.divide(tf.reduce_sum(self.discriminator_grads[0][0]**2), 
                        tf.reduce_sum(self.discriminator_grads[0][1]**2)))

            discriminator_summaries = tf.summary.merge([discriminator_loss_A_summary, 
                discriminator_loss_B_summary, discriminator_loss_summary, 
                discriminator_gradient_summary])

        return predictor_summaries, discriminator_summaries


if __name__ == '__main__':
    
    model = VariationalCycleGAN(dim_pitch=1, dim_mfc=23)
    print('Graph Compile Successful.')
