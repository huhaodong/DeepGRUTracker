import tensorflow as tf
import util.loadData as loadData
import model.deepGRUTracker as deepGRUTracker
import os

def train(opt):
    ''' train deep GRU tracker '''
    maxTargetNumber = opt.max_target_number
    detDim = opt.det_dim
    # gruHideSize = opt.gru_hide_size
    productDim = opt.product_dim
    # productHiddenSize = opt.product_hidden_size
    inputWSize = opt.input_w_size
    inputHSize = opt.input_h_size
    # batchSize = opt.batch_size
    # trainPath = opt.train_path
    epoch = opt.epoch
    modelSavePath = opt.model_save_path
    modelSaveEpoch = opt.model_save_epoch
    summaryLogSavePath = opt.summary_log_save_path
    # featrueHiddenSize = 34*60*512

    inputImage = tf.placeholder(tf.float32,shape=[None,inputHSize,inputWSize,3],name='input_images')
    inputDet = tf.placeholder(tf.float32,shape=[None,maxTargetNumber,detDim],name='input_det')
    inputGt = tf.placeholder(tf.float32,shape=[None,maxTargetNumber,productDim],name='gt')

    modelArgs = {'input_image':inputImage,'input_det':inputDet,'superOpt':opt}
    tracker = deepGRUTracker.DeepGRUTracker(modelArgs)

    loss = tf.reduce_mean(tf.pow(tf.subtract(inputGt,tracker.hid_6),2.0))
    tf.summary.scalar("loss_function",loss)

    optimiz = tf.train.AdamOptimizer(0.04)
    train = optimiz.minimize(loss)

    init = tf.global_variables_initializer()

    tfconfig = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)

    saver = tf.train.Saver(max_to_keep=3)

    with tf.Session(config=tfconfig) as sess:
        with tf.device("/gpu:1"):
            
            sess.run(init)

            merged_summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(summaryLogSavePath,sess.graph)

            for i in range(1,epoch+1):
                dataLoder = loadData.DataLoader(opt)
                dataLoder.flashLoader()
                data = dataLoder.next()
                loss = 0
                while True:
                    if data['img'] ==None or data['det']==None:
                        break
                    _,loss = sess.run([train,loss],feed_dict={inputImage:data['img'],inputDet:data['det'],inputGt:data['gt']})
                    summary_str = sess.run(merged_summary_op,feed_dict={inputImage:data['img'],inputDet:data['det'],inputGt:data['gt']})
                    summary_writer.add_summary(summary_str,i)
                    data = dataLoder.next()
                    
                dataLoder.endLoader()
                if i%modelSaveEpoch==0:
                    savePath = os.path.join(modelSavePath,"deep_GRU_tracker.cpkt")
                    saver.save(sess,savePath,global_step=i)
                    print('eopch=',str(i))