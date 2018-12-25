import tensorflow as tf
import util.loadData as loadData
import model.deepGRUTracker as deepGRUTracker
import os
import numpy as np
import util.dataWriter as dataWriter


def test(opt):
    ''' train deep GRU tracker '''
    maxTargetNumber = opt.max_target_number
    detDim = opt.det_dim
    # gruHideSize = opt.gru_hide_size
    # productHiddenSize = opt.product_hidden_size
    inputWSize = opt.img_resize_w
    inputHSize = opt.img_resize_h
    # batchSize = opt.batch_size
    # trainPath = opt.train_path
    loadEpoch = opt.load_epoch
    modelSavePath = opt.model_save_path
    allFeatureDim = opt.all_feature_dim
    sequenceSize = opt.sequence_size
    bantchSize = opt.batch_size
    trackerResultPath = opt.tracker_result_path
    testLoopSize = opt.test_loop_size
    picFeatureDim = opt.pic_feature_dim
    detFeatureDim= opt.det_feature_dim
    gpuIndex = opt.gpu_index
    # featrueHiddenSize = 34*60*512

    inputImage = tf.placeholder(
        tf.float32, shape=[None, inputHSize, inputWSize, 3], name='input_images')
    inputDet = tf.placeholder(
        tf.float32, shape=[None, maxTargetNumber, detDim], name='input_det')

    # inputGt = tf.placeholder(
    #     tf.float32, shape=[None, maxTargetNumber, productDim], name='gt')

    fuseSequence = tf.placeholder(
        tf.float32, shape=[None, sequenceSize, allFeatureDim], name='fuse_sequence')
    
    picFeatureSequence = tf.placeholder(
        tf.float32, shape=[None, sequenceSize, picFeatureDim], name='pic_feature_sequence')

    detFeatureSequence = tf.placeholder(
        tf.float32, shape=[None, sequenceSize, detFeatureDim], name='det_feature_sequence')

    modelArgs = {
        'input_image': inputImage,
        'input_det': inputDet,
        'sequence': fuseSequence,
        'picFeatureSequence': picFeatureSequence,
        'detFeatureSequence': detFeatureSequence,
        'superOpt': opt
    }

    tracker = deepGRUTracker.DeepGRUTracker(modelArgs)

    init = tf.global_variables_initializer()

    tfconfig = tf.ConfigProto(
        log_device_placement=False, allow_soft_placement=True)

    saver = tf.train.Saver()

    with tf.Session(config=tfconfig) as sess:
        with tf.device("/gpu:"+str(gpuIndex)):

            sess.run(init)
            if loadEpoch != 0:
                saver.restore(sess, os.path.join(
                    modelSavePath, "deep_GRU_tracker.cpkt-"+str(loadEpoch)))
            dataLoder = loadData.DataLoader(opt)
            for i in range(testLoopSize):
                dataLoder.flashLoader()
                data = dataLoder.next()
                # sequence = tf.Variable(tf.zeros([bantchSize,sequenceSize,allFeatureDim]),tf.float32)
                sequence = np.zeros((bantchSize, sequenceSize, allFeatureDim))
                detsequence = np.zeros((bantchSize, sequenceSize, detFeatureDim))
                picsequence = np.zeros((bantchSize, sequenceSize, picFeatureDim))

                resultMap = {}
                frameIndex = 0
                while True:
                    frameIndex += 1
                    if data['img'] == [] or data['det'] == []:
                        break
                    hid_6, hid_4, hid_det, hid_pic = sess.run([tracker.hid_6, tracker.hid_4, tracker.hid_fc_1, tracker.hid_fc_2], 
                                                                feed_dict={inputImage: data['img'],
                                                                          inputDet: data['det'],
                                                                          fuseSequence: sequence,
                                                                          picFeatureSequence: picsequence,
                                                                          detFeatureSequence: detsequence
                                                                          })
                    # new sequence
                    # _,keepSeq = tf.split(1,sequence,[1,sequenceSize-1])
                    # sequence = tf.concat(axis=1,values=[keepSeq,hid9])

                    resultMap[frameIndex] = tf.floor(hid_6[0])

                    seqlist = np.split(sequence, [1], 1)
                    detseqlist = np.split(detsequence, [1], 1)
                    picseqlist = np.split(picsequence, [1], 1)

                    # hid9Numpy = hid9.eval(session=sess)
                    sequence = np.concatenate((seqlist[-1], hid_4), axis=1)
                    detsequence = np.concatenate((detseqlist[-1], hid_det), axis=1)
                    picsequence = np.concatenate((picseqlist[-1], hid_pic), axis=1)
                    
                    data = dataLoder.next()
                dataWriter.writerTrackerResult(resultMap,trackerResultPath,str(i)+'.txt')
                resultMap.clear()
                dataLoder.endLoader()
