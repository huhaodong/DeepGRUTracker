import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow import nn


def deepGRUNet(
    input, batchSize, _scopeName, is_train=False, keep_prob=0.5, _reuseFlage=True, n_layer=1, n_hidden=128
):
    ''' return a deep GRU net for product the bboxs'''

    stacked_rnn = []
    # with tf.variable_scope(_scopeName,reuse=_reuseFlage) as scope:
    for _ in range(n_layer):
        grucell = rnn.GRUCell(n_hidden)
        if is_train:
            grucell = nn.rnn_cell.DropoutWrapper(
                grucell, output_keep_prob=keep_prob)
        stacked_rnn.append(grucell)
    mcell = rnn.MultiRNNCell(stacked_rnn)
    initial_state = mcell.zero_state(batchSize, tf.float32)
    output, _ = nn.dynamic_rnn(
        mcell,
        input,
        initial_state=initial_state,
        dtype=tf.float32)  # get deep GRU Net output
    output = tf.transpose(output, [1, 0, 2])
    return output
