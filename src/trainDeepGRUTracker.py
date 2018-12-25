import tensorflow as tf
import util.loadData as loadData
import model.deepGRUTracker as deepGRUTracker
import os
import numpy as np


def train(opt):
    ''' train deep GRU tracker '''
    maxTargetNumber = opt.max_target_number
    detDim = opt.det_dim
    # gruHideSize = opt.gru_hide_size
    productDim = opt.product_dim
    # productHiddenSize = opt.product_hidden_size
    inputWSize = opt.img_resize_w
    inputHSize = opt.img_resize_h
    # batchSize = opt.batch_size
    # trainPath = opt.train_path
    epoch = opt.epoch
    loadEpoch = opt.load_epoch
    modelSavePath = opt.model_save_path
    modelSaveEpoch = opt.model_save_epoch
    summaryLogSavePath = opt.summary_log_save_path
    allFeatureDim = opt.all_feature_dim
    sequenceSize = opt.sequence_size
    bantchSize = opt.batch_size
    learnRate = opt.learn_rate
    gpuIndex = opt.gpu_index
    # featrueHiddenSize = 34*60*512

    inputImage = tf.placeholder(
        tf.float32, shape=[None, inputHSize, inputWSize, 3], name='input_images')
    inputDet = tf.placeholder(
        tf.float32, shape=[None, maxTargetNumber, detDim], name='input_det')
    inputGt = tf.placeholder(
        tf.float32, shape=[None, maxTargetNumber, productDim], name='gt')

    tmpSequence = tf.placeholder(
        tf.float32, shape=[None, sequenceSize, allFeatureDim], name='tmp_sequence')

    modelArgs = {
        'input_image': inputImage,
        'input_det': inputDet,
        'sequence': tmpSequence,
        'superOpt': opt
    }

    tracker = deepGRUTracker.DeepGRUTracker(modelArgs)

    loss = tf.reduce_mean(tf.pow(tf.subtract(inputGt, tracker.hid_6), 2.0))
    tf.summary.scalar("loss_function", loss)

    optimiz = tf.train.AdamOptimizer()
    train = optimiz.minimize(loss)

    init = tf.global_variables_initializer()

    tfconfig = tf.ConfigProto(
        log_device_placement=True, allow_soft_placement=True)

    saver = tf.train.Saver(max_to_keep=3)
    with tf.Session(config=tfconfig) as sess:
        with tf.device("/gpu:"+str(gpuIndex)):

            sess.run(init)
            if loadEpoch != 0:
                saver.restore(sess, os.path.join(
                    modelSavePath, "deep_GRU_tracker.cpkt-"+str(loadEpoch)))
            merged_summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(
                summaryLogSavePath, sess.graph)
            loopEnd = epoch+1
            loopStart = loadEpoch+1
            dataLoder = loadData.DataLoader(opt)
            for i in range(loopStart, loopEnd):
                dataLoder.flashLoader()
                data = dataLoder.next()
                # sequence = tf.Variable(tf.zeros([bantchSize,sequenceSize,allFeatureDim]),tf.float32)
                sequence = np.zeros((bantchSize, sequenceSize, allFeatureDim))
                while True:
                    if data['img'] == [] or data['det'] == []:
                        break
                    _, hid_4 = sess.run([train, tracker.hid_4], feed_dict={inputImage: data['img'],
                                                                          inputDet: data['det'],
                                                                          inputGt: data['gt'],
                                                                          tmpSequence: sequence
                                                                          })

                    summary_str = sess.run(merged_summary_op, feed_dict={inputImage: data['img'],
                                                                         inputDet: data['det'],
                                                                         inputGt: data['gt'],
                                                                         tmpSequence: sequence
                                                                         })
                    summary_writer.add_summary(summary_str, i)

                    # new sequence
                    # _,keepSeq = tf.split(1,sequence,[1,sequenceSize-1])
                    # sequence = tf.concat(axis=1,values=[keepSeq,hid9])
                    seqlist = np.split(sequence, [1], 1)
                    # hid9Numpy = hid9.eval(session=sess)
                    sequence = np.concatenate((seqlist[-1], hid_4), axis=1)

                    data = dataLoder.next()

                dataLoder.endLoader()
                print('epoch:', str(i), " finished!")
                if i % modelSaveEpoch == 0:
                    savePath = os.path.join(
                        modelSavePath, "deep_GRU_tracker.cpkt")
                    saver.save(sess, savePath, global_step=i)
                    # print('epoch=',str(i))
