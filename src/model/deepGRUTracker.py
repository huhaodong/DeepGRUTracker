import tensorflow as tf 
import model.vgg16 as vgg16
import model.deepGRU as deepGRU

class DeepGRUTracker:

    def __init__(self,opt):

        inputImage = opt["input_image"]
        inputDet = opt["input_det"]
        inputSequence = opt["sequence"] #shape [batchSize*(sequenceSize-1)*allFeatureDime]
        maxTargetNumber = opt["superOpt"].max_target_number
        detDim = opt["superOpt"].det_dim
        picFeatureDim = opt["superOpt"].pic_feature_dim
        detFeatureDim = opt["superOpt"].det_feature_dim
        allFeatureDim = opt["superOpt"].all_feature_dim
        inputHSize = opt["superOpt"].input_h_size
        inputWSize = opt["superOpt"].input_w_size
        vgg16NpyPath = opt["superOpt"].vgg16_npy_path
        gruHideSize = opt["superOpt"].gru_hide_size
        productDim = opt["superOpt"].product_dim
        fcSecHiddenSize = opt["superOpt"].fc_sec_hidden_size
        gruKeepProb = opt["superOpt"].gru_keep_prob
        isTrain = opt["superOpt"].is_train

        with tf.variable_scope('deep_gru_tracker'):
            # fc1 in order to weight the input det
            self.x_det = tf.reshape(inputDet,[-1,maxTargetNumber*detDim],name='reshape_1')  #shape [batch_size*(max_target_number*det_dim)]
            self.fc_1 = tf.layers.dense(self.x_det,detFeatureDim,activation=tf.nn.relu,name='fc_1') #shape [batch_size*detFeatureDim]

            # self.hid_1 = tf.reshape(self.fc_1,[-1,inputHSize,inputWSize,1],name='reshap_fc_1')
            # self.hid_1 = tf.concat([self.hid_1,self.hid_1,self.hid_1],3,name="concat_hid_1")    #shape [batch_size*input_h_size*input_w_size*3]

            # add fc1 and input image
            # self.hid_2= tf.add(inputImage,self.hid_1,name='add_1')  #shape [batch_size*input_h_size*input_w_size*3]
            
            # get feature map using vgg16
            self.vgg16net = vgg16.Vgg16(vgg16_npy_path=vgg16NpyPath)
            self.vgg16net.build(inputImage)     #inputImage shap [batch_size*512*512*3]
            self.hid_1 = self.vgg16net.pool5    #shape [batch_size*16*16*512]

            # conv1
            self.conv1_1 = self.conv_layer(self.hid_1,[3,3,512,1024],'conv1_1')
            self.conv1_2 = self.conv_layer(self.conv1_1,[1,1,1024,1024],'conv1_2')
            self.pool_1 = self.max_pool(self.conv1_2,'pool_1')  #shape [batch_size*4*4*1024]

            # fc2
            featchMapDim = 1
            featchMapShape = self.hid_1.get_shape().as_list()
            for d in featchMapShape[1:]:
                featchMapDim *= d
            self.hid_2 = tf.reshape(self.pool_1,[-1,featchMapDim],name='reshape_2')
            self.fc_2 = tf.layers.dense(self.hid_2,picFeatureDim,activation=tf.nn.relu,name='fc_2') #shape [batch_size*picFeatureDim]

            # normalization
            self.nor_1 = tf.nn.l2_normalize(self.fc_2, dim=1)
            self.nor_2 = tf.nn.l2_normalize(self.fc_1, dim=1)

            # concat dim feature and pic feature
            self.hid_3 = tf.concat(1,[self.nor_1,self.nor_2], name='concat_1')  #shape [batch_size*(picFeatureDim+detFeatureDim)]

            # fc3
            self.fc_3 = tf.layers.dense(self.hid_3,allFeatureDim,activation=tf.nn.relu,name='fc_3') #shape [batch_size*allFeatureDim]
            self.hid_4 = tf.reshape(self.fc_3,[-1,1,allFeatureDim])     #shape [batch_size*1*allFeatureDim]

            # concat sequence
            self.hid_5 = tf.concat(1,[inputSequence,self.fc_3], name='concat_2')  #shape [batch_size*sequenceSize*allFeatureDim]

            # deep GRU net
            # self.hid_4 = tf.reshape(self.hid_5,[-1,maxTargetNumber,gruHideSize],name='hid_4') #shape [batch_size*max_target_number*gru_hide_size]
            self.gru_1 = deepGRU.deepGRUNet(self.hid_5,_scopeName='deep_gru',is_train=isTrain,keep_prob=gruKeepProb,n_layer=1,n_hidden=gruHideSize) #shape [sequenceSize*batch_size*gru_hide_size]

            # fc4
            # self.hid_5 = tf.reshape(self.gru_1[-1],[-1,maxTargetNumber*gruHideSize],name='hid_5')   #shape [batch_size*(max_target_number*gru_hide_size)]
            self.fc_4 = tf.layers.dense(self.gru_1[-1],maxTargetNumber*fcSecHiddenSize,activation=tf.nn.relu,name='fc_4') #shape [batch_size*(maxTargetNumber*fcSecHiddenSize)]

            # fc5
            self.fc_5 = tf.layers.dense(self.fc_4,maxTargetNumber*productDim,activation=tf.nn.relu,name='fc_5')    #shape [batch_size*(maxTargetNumber*productDim)]
            self.hid_6 = tf.reshape(self.fc_5,[-1,maxTargetNumber,productDim],name="hid_6")  #shape [batch_size*maxTargetNumber*productDim] product output

            # loop fc6
            self.hid_7 = tf.reshape(self.hid_6,[-1,maxTargetNumber*productDim],name='hid_7')  #shape [batch_size*(max_target_number*productDim)]
            self.fc_6 = tf.layers.dense(self.hid_7,detFeatureDim,activation=tf.nn.relu,name='fc_6') #shape [batch_size*detFeatureDim]

            # loop product normalize
            self.nor_3 = tf.nn.l2_normalize(self.fc_6, dim=1)

            # concat pic feature map and product feature
            self.hid_8 = tf.concat(1,[self.nor_1,self.nor_3], name='hid_8')  #shape [batch_size*(picFeatureDim+detFeatureDim)]

            # loop fc7
            self.fc_7 = tf.layers.dense(self.hid_8,allFeatureDim,activation=tf.nn.relu,name='fc_7') #shape [batch_size*allFeatureDim]
            self.hid_9 = tf.reshape(self.fc_7,[-1,1,allFeatureDim])     #shape [batch_size*1*allFeatureDim] loop output

    def conv_layer(self, bottom, filtShapfc_2
        with tf.variable_scope(name,reuse=True):
            filt = tf.get_variable("filt",shape=filtShape,
                initializer=tf.random_normal_initializer(dtype=tf.float32),
                dtype=tf.float32)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = tf.get_variable("biases",shape=[filtShape[-1]],
                initializer=tf.random_normal_initializer(dtype=tf.float32),
                dtype=tf.float32)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)