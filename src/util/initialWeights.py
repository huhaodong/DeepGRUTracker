import tensorflow as tf 

def convWeightVariable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def convBiasVariable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def fcWeightVariable(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)
    
def fcBiasVariable(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)