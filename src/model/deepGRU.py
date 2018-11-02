import tensorflow as tf 
from tensorflow.contrib import rnn
from tensorflow import nn

def deepGRUNet(
    input
    ,_scopeName
    ,_reuseFlage=True
    ,n_layer=1
    ,n_hidden=128
    ):
    
    ''' return a deep GRU net for product the bboxs'''

    stacked_rnn=[]
    with tf.variable_scope(_scopeName,reuse=_reuseFlage) as scope:
        for _ in range(n_layer):
            stacked_rnn.append(rnn.GRUCell(n_hidden))
        mcell = rnn.MultiRNNCell(stacked_rnn)
        output,_=nn.dynamic_rnn(mcell,input,dtype=tf.float64) # get deep GRU Net output

    return output