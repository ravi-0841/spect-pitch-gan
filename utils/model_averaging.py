import tensorflow as tf
session_unif = tf.Session()
session_gend = tf.Session()
session_avg = tf.Session()


with session_gend as sess:
    loader1 = tf.train.import_meta_graph('/home/ravi/Desktop/sum_mfc_models/neu-ang/lp_0.0001_le_1e-06_li_0.0_lrg_1e-05_lrd_1e-07_sum_mfc_gender_shuffled/neu-ang_200.ckpt.meta')
    loader1.restore(sess, '/home/ravi/Desktop/sum_mfc_models/neu-ang/lp_0.0001_le_1e-06_li_0.0_lrg_1e-05_lrd_1e-07_sum_mfc_gender_shuffled/neu-ang_200.ckpt')
    var_gend = sess.run(tf.trainable_variables())

with session_unif as sess:
    loader2 = tf.train.import_meta_graph('/home/ravi/Desktop/sum_mfc_models/neu-ang/lp_1e-05_le_0.1_li_0.0_lrg_1e-05_lrd_1e-07_sum_mfc_best/neu-ang_200.ckpt.meta')
    loader2.restore(sess, '/home/ravi/Desktop/sum_mfc_models/neu-ang/lp_1e-05_le_0.1_li_0.0_lrg_1e-05_lrd_1e-07_sum_mfc_best/neu-ang_200.ckpt')
    var_unif = sess.run(tf.trainable_variables())



