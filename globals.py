import tensorflow as tf
def initialize():
	global global_step
	global_step = tf.Variable(-1)