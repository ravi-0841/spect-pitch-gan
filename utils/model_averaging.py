import tensorflow as tf
import numpy as np


checkpoints = ['/home/ravi/Desktop/mixing_models/lp_1e-05_le_0.1_li_0.0_lrg_1e-05_lrd_1e-07_sum_mfc_gender_neu-ang_random_seed_5/neu-ang_200.ckpt', \
               '/home/ravi/Desktop/sum_mfc_models/neu-ang/lp_1e-05_le_0.1_li_0.0_lrg_1e-05_lrd_1e-07_sum_mfc_epoch_200_best/neu-ang_200.ckpt']

weights = {}
weights[checkpoints[0]] = 0.2
weights[checkpoints[1]] = 0.8

tf.logging.info("Reading variables and averaging checkpoints:")
for c in checkpoints:
    tf.logging.info("%s ", c)
var_list = tf.train.list_variables(checkpoints[0])
var_values, var_dtypes = {}, {}
for (name, shape) in var_list:
    if not name.startswith("global_step"):
        var_values[name] = np.zeros(shape)
for checkpoint in checkpoints:
    reader = tf.train.load_checkpoint(checkpoint)
    for name in var_values:
        tensor = reader.get_tensor(name)
        var_dtypes[name] = tensor.dtype
        var_values[name] += weights[checkpoint]*tensor
    tf.logging.info("Read from checkpoint %s", checkpoint)

with tf.variable_scope(tf.get_variable_scope(), 
                       reuse=tf.AUTO_REUSE):
    tf_vars = [tf.get_variable(v, shape=var_values[v].shape, 
                               dtype=var_dtypes[v]) for v in var_values]

placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
saver = tf.train.Saver(tf.all_variables())

# Build a model consisting only of variables, set them to the average values.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for p, assign_op, (name, value) in zip(placeholders, assign_ops, var_values.items()):
        sess.run(assign_op, {p: value})
        print('Assigned: ' + name)

    # Use the built saver to save the averaged checkpoint.
    saver.save(sess, '/home/ravi/Desktop/mixing_models/averaged_2_8.ckpt')

print("Averaged checkpoints saved")