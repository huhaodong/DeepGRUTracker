import tensorflow as tf 
import util.initialWeights as initialWeights
import util.loadData as loadData
import model.deepGRUTracker as deepGRUTracker

def train(opt):
    ''' train deep GRU tracker '''
    maxTargetNumber = opt.max_target_number
    detDim = opt.det_dim
    gruHideSize = opt.gru_hide_size
    productDim = opt.product_dim
    productHiddenSize = opt.product_hidden_size
    inputWSize = opt.input_w_size
    inputHSize = opt.input_h_size
    batchSize = opt.batch_size
    trainPath = opt.train_path
    epoch = opt.epoch
    featrueHiddenSize = 34*60*512

    # weights = {
    #     'fc_det_1':
    #     initialWeights.fcWeightVariable([
    #         input_h_size
    #         ,max_target_number
    #         ]),
    #     'fc_det_2':
    #     initialWeights.fcWeightVariable([
    #         det_dim
    #         ,input_w_size
    #     ]),
    #     'fc_featrue_1':
    #     initialWeights.fcWeightVariable([
    #         featrue_hidden_size
    #         ,max_target_number*gru_hide_size
    #     ]),
    #     'fc_output_1':
    #     initialWeights.fcWeightVariable([
    #         gru_hide_size
    #         ,product_hidden_size
    #     ]),
    #     'fc_output_2':
    #     initialWeights.fcWeightVariable([
    #         product_hidden_size
    #         ,product_dim
    #     ])
    # }
    # biases = {
    #     'fc_det_1':
    #     initialWeights.fcBiasVariable([
    #         max_target_number
    #         ]),
    #     'fc_det_2':
    #     initialWeights.fcBiasVariable([
    #         input_w_size
    #     ]),
    #     'fc_featrue_1':
    #     initialWeights.fcBiasVariable([
    #         max_target_number*gru_hide_size
    #     ]),
    #     'fc_output_1':
    #     initialWeights.fcBiasVariable([
    #         product_hidden_size
    #     ]),
    #     'fc_output_2':
    #     initialWeights.fcWeightVariable([
    #         product_dim
    #     ])
    # }

    inputImage = tf.placeholder(tf.float32,shape=[None,inputHSize,inputWSize],name='input_images')
    inputDet = tf.placeholder(tf.float32,shape=[None,maxTargetNumber,detDim],name='input_det')
    inputGt = tf.placeholder(tf.float32,shape=[None,maxTargetNumber,productDim],name='gt')

    modelArgs = {'input_image':inputImage,'input_det':inputDet,'superOpt':opt}
    tracker = deepGRUTracker.DeepGRUTracker(modelArgs)
    with tf.Session() as sess:
        for i in range(epoch):
            imgs,det,gt = loadData.loadTrainData(batchSize,trainPath)