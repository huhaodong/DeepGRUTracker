import tensorflow as tf
import util.loadData as loadData
import model.deepGRUTracker as deepGRUTracker
import os
import numpy as np
import util.dataWriter as dataWriter
import util.show as show

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
    gpuIndex = opt.gpu_index
    # featrueHiddenSize = 34*60*512
    featureMapOutputPath = "F:\\work\\experiment\\deepGruTracker\\result\\v1.0.0\\featureMap\\MOT16-09-pool2"

    inputImage = tf.placeholder(
        tf.float32, shape=[None, inputHSize, inputWSize, 3], name='input_images')
    inputDet = tf.placeholder(
        tf.float32, shape=[None, maxTargetNumber, detDim], name='input_det')

    # inputGt = tf.placeholder(
    #     tf.float32, shape=[None, maxTargetNumber, productDim], name='gt')

    tmpSequence = tf.placeholder(
        tf.float32, shape=[None, sequenceSize, allFeatureDim], name='tmp_sequence')

    modelArgs = {
        'input_image': inputImage,
        'input_det': inputDet,
        'sequence': tmpSequence,
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
                resultMap = {}
                frameIndex = 0
                imageCountor = 1

                while True:
                    frameIndex += 1
                    if data['img'] == [] or data['det'] == []:
                        break
                    hid_6, hid_4, oriFeature = sess.run([tracker.hid_6,tracker.hid_4 ,tracker.pool_2], feed_dict={inputImage: data['img'],
                                                                          inputDet: data['det'],
                                                                          tmpSequence: sequence
                                                                          })
                    # new sequence
                    # _,keepSeq = tf.split(1,sequence,[1,sequenceSize-1])
                    # sequence = tf.concat(axis=1,values=[keepSeq,hid9])

                    resultMap[frameIndex] = tf.floor(hid_6[0])

                    seqlist = np.split(sequence, [1], 1)
                    # hid9Numpy = hid9.eval(session=sess)
                    sequence = np.concatenate((seqlist[-1], hid_4), axis=1)

                    # save the feature map picture
                    featureMap = oriFeature[0]
                    orImage = data['img'][0]
                    fmOutputPath = os.path.join(featureMapOutputPath,str(imageCountor)+'.png')
                    show.saveFeatureMap(img=orImage ,featureMap=featureMap ,outputPath=fmOutputPath)

                    data = dataLoder.next()
                    imageCountor += 1

                dataWriter.writerTrackerResult(resultMap,trackerResultPath,str(i)+'.txt')
                resultMap.clear()
                dataLoder.endLoader()
