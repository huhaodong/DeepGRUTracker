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
        batchSize = opt["superOpt"].batch_size
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
        nnKeepProb = opt["superOpt"].nn_keep_prob
        gruLarys = opt["superOpt"].gru_larys
        gpuIndex = opt["superOpt"].gpu_index
        self.gpuIndex = gpuIndex
        with tf.variable_scope('deep_gru_tracker'):
            with tf.device("/gpu:"+str(gpuIndex)):
                # fc1 in order to weight the input det
                self.x_det = tf.reshape(inputDet,[-1,maxTargetNumber*detDim],name='reshape_1')  #shape [batch_size*(max_target_number*det_dim)]
                self.fc_1 = tf.layers.dense(self.x_det,detFeatureDim,name='fc_1') #shape [batch_size*detFeatureDim]
                self.fc_1_nor = self.normalization_layer(self.fc_1,detFeatureDim,name="fc_1_nor")
                self.fc_1_relu = tf.nn.relu(self.fc_1_nor)
                self.fc_1 = tf.nn.dropout(self.fc_1_relu,nnKeepProb)
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
                self.pool_1 = self.max_pool(self.conv1_2,'pool_1')  #shape [batch_size*8*8*1024]

                # conv2
                self.conv2_1 = self.conv_layer(self.pool_1,[3,3,1024,1024],'conv2_1')
                self.conv2_2 = self.conv_layer(self.conv2_1,[1,1,1024,1024],'conv2_2')
                self.pool_2 = self.max_pool(self.conv2_2,'pool_2')  #shape [batch_size*4*4*1024]

                # fc2
                featchMapDim = 1
                featchMapShape = self.pool_2.get_shape().as_list()
                for d in featchMapShape[1:]:
                    featchMapDim *= d
                self.hid_2 = tf.reshape(self.pool_2,[-1,featchMapDim],name='reshape_2')
                self.fc_2 = tf.layers.dense(self.hid_2,picFeatureDim,name='fc_2') #shape [batch_size*picFeatureDim]
                self.fc_2_nor = self.normalization_layer(self.fc_2,picFeatureDim,name="fc_2_nor")
                self.fc_2_relu = tf.nn.relu(self.fc_2_nor)
                self.fc_2 = tf.nn.dropout(self.fc_2_relu,nnKeepProb)

                # normalization
                self.nor_1 = tf.nn.l2_normalize(self.fc_2, axis=1)
                self.nor_2 = tf.nn.l2_normalize(self.fc_1, axis=1)
                
                self.nor_1_shape = self.nor_1.get_shape().as_list()
                self.nor_2_shape = self.nor_2.get_shape().as_list()

                # concat dim feature and pic feature
                self.hid_3 = tf.concat(values=[self.nor_1,self.nor_2], axis=1, name='concat_1')  #shape [batch_size*(picFeatureDim+detFeatureDim)]
                self.hid_3_shape = self.hid_3.get_shape().as_list()

                # fc3
                self.fc_3 = tf.layers.dense(self.hid_3,allFeatureDim,name='fc_3') #shape [batch_size*allFeatureDim]
                self.fc_3_nor = self.normalization_layer(self.fc_3,allFeatureDim,name="fc_3_nor")
                self.fc_3_relu = tf.nn.relu(self.fc_3_nor)
                self.fc_3 = tf.nn.dropout(self.fc_3_relu,nnKeepProb)

                self.fc_3_shape = self.fc_3.get_shape().as_list()

                self.hid_4 = tf.reshape(self.fc_3,[-1,1,allFeatureDim])     #shape [batch_size*1*allFeatureDim]

                # concat sequence
                self.hid_5 = tf.concat(values=[inputSequence,self.hid_4],axis=1, name='concat_2')  #shape [batch_size*sequenceSize*allFeatureDim]
                # self.hid_5_shape = self.hid_5.get_shape().as_list()
                # deep GRU net
                # self.hid_4 = tf.reshape(self.hid_5,[-1,maxTargetNumber,gruHideSize],name='hid_4') #shape [batch_size*max_target_number*gru_hide_size]
                self.gru_1 = deepGRU.deepGRUNet(
                    self.hid_5,
                    batchSize=batchSize,
                    _scopeName='deep_gru',
                    is_train=isTrain,
                    keep_prob=gruKeepProb,
                    n_layer=gruLarys,
                    n_hidden=gruHideSize
                    ) #shape [sequenceSize*batch_size*gru_hide_size]

                # fc4
                # self.hid_5 = tf.reshape(self.gru_1[-1],[-1,maxTargetNumber*gruHideSize],name='hid_5')   #shape [batch_size*(max_target_number*gru_hide_size)]
                self.fc_4 = tf.layers.dense(self.gru_1[-1],maxTargetNumber*fcSecHiddenSize,name='fc_4') #shape [batch_size*(maxTargetNumber*fcSecHiddenSize)]
                self.fc_4_nor = self.normalization_layer(self.fc_4,maxTargetNumber*fcSecHiddenSize,name="fc_4_nor")
                self.fc_4_relu = tf.nn.relu(self.fc_4_nor)
                self.fc_4 = tf.nn.dropout(self.fc_4_relu,nnKeepProb)

                # fc5
                self.fc_5 = tf.layers.dense(self.fc_4,maxTargetNumber*productDim,name='fc_5')    #shape [batch_size*(maxTargetNumber*productDim)]
                self.hid_6 = tf.reshape(self.fc_5,[-1,maxTargetNumber,productDim],name="hid_6")  #shape [batch_size*maxTargetNumber*productDim] product output

                # # loop fc6
                # self.hid_7 = tf.reshape(self.hid_6,[-1,maxTargetNumber*productDim],name='hid_7')  #shape [batch_size*(max_target_number*productDim)]
                # self.fc_6 = tf.layers.dense(self.hid_7,detFeatureDim,activation=tf.nn.relu,name='fc_6') #shape [batch_size*detFeatureDim]
                # self.fc_6 = tf.nn.dropout(self.fc_6,nnKeepProb)
                
                # # loop product normalize
                # self.nor_3 = tf.nn.l2_normalize(self.fc_6, axis=1)

                # # concat pic feature map and product feature
                # self.hid_8 = tf.concat(values=[self.nor_1,self.nor_3],axis=1, name='hid_8')  #shape [batch_size*(picFeatureDim+detFeatureDim)]

                # # loop fc7
                # self.fc_7 = tf.layers.dense(self.hid_8,allFeatureDim,activation=tf.nn.relu,name='fc_7') #shape [batch_size*allFeatureDim]
                
                # self.hid_9 = tf.reshape(self.fc_7,[-1,1,allFeatureDim])     #shape [batch_size*1*allFeatureDim] loop output

    def conv_layer(self, bottom, filtShape, name):
        with tf.variable_scope(name):
            
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

    def normalization_layer(self, input, out_dime, name=None, axes=[0], epsilon=0.001):
        with tf.variable_scope(name):
            with tf.device("/gpu:"+str(self.gpuIndex)):
                fc_mean, fc_var = tf.nn.moments(
                    input,
                    axes=axes
                )
                scale = tf.Variable(tf.ones([out_dime]))
                shift = tf.Variable(tf.zeros([out_dime]))
                output = tf.nn.batch_normalization(input, fc_mean, fc_var, shift, scale, epsilon)
                return output

    def max_pool(self, bottom, name):
        ret = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
        return ret