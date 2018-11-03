import tensorflow as tf 
import model.vgg16 as vgg16
import model.deepGRU as deepGRU

class DeepGRUTracker:

    def __init__(self,opt):

        inputImage = opt["input_image"]
        inputDet = opt["input_det"]
        maxTargetNumber = opt["superOpt"].max_target_number
        detDim = opt["superOpt"].det_dim
        inputHSize = opt["superOpt"].input_h_size
        inputWSize = opt["superOpt"].input_w_size
        vgg16NpyPath = opt["superOpt"].vgg16_npy_path
        gruHideSize = opt["superOpt"].gru_hide_size
        product_dim = opt["superOpt"].product_dim
        fc3HiddenSize = opt["superOpt"].fc_3_hidden_size
        gruKeepProb = opt["superOpt"].gru_keep_prob
        isTrain = opt["superOpt"].is_train

        # fc1 in order to weight the input det
        self.x_det = tf.reshape(inputDet,[-1,maxTargetNumber*detDim],name='reshape_det_dim')  #shape [batch_size*(max_target_number*det_dim)]
        self.fc_1 = tf.layers.dense(self.x_det,inputHSize*inputWSize,activation=tf.nn.relu,name='fc_1') #shape [batch_size*(inputHSize*inputWSize)]
        self.hid_1 = tf.reshape(self.fc_1,[-1,inputHSize,inputWSize,1],name='reshap_fc_1')
        self.hid_1 = tf.concat([self.hid_1,self.hid_1,self.hid_1],3,name="concat_hid_1")    #shape [batch_size*input_h_size*input_w_size*3]

        # add fc1 and input image
        self.hid_2= tf.add(inputImage,self.hid_1,name='add_1')  #shape [batch_size*input_h_size*input_w_size*3]
        
        # get featch map using vgg16
        self.vgg16net = vgg16.Vgg16(vgg16_npy_path=vgg16NpyPath)
        self.vgg16net.build(self.hid_2)
        self.hid_3 = self.vgg16net.pool5    #shape [batch_size*34*60*512]

        # fc2
        featchMapDim = 1
        featchMapShape = self.hid_3.get_shape().as_list()
        for d in featchMapShape[1:]:
            featchMapDim *= d
        self.hid_3_c = tf.reshape(self.hid_3,[-1,featchMapDim],name='reshape_featch_map')
        self.fc_2 = tf.layers.dense(self.hid_3_c,maxTargetNumber*gruHideSize,activation=tf.nn.relu,name='fc_2') #shape [batch_size*(max_target_number*gru_hide_size)]

        # deep GRU net
        self.hid_4 = tf.reshape(self.fc_2,[-1,maxTargetNumber,gruHideSize],name='hid_4') #shape [batch_size*max_target_number*gru_hide_size]
        self.hid_5 = deepGRU.deepGRUNet(self.hid_4,_scopeName='deep_gru',is_train=isTrain,keep_prob=gruKeepProb,n_layer=1,n_hidden=gruHideSize) #shape [batch_size*max_target_number*gru_hide_size]

        # fc3
        self.hid_5 = tf.reshape(self.hid_5,[-1,maxTargetNumber*gruHideSize],name='hid_5')   #shape [batch_size*(max_target_number*gru_hide_size)]
        self.fc_3 = tf.layers.dense(self.hid_5,maxTargetNumber*fc3HiddenSize,activation=tf.nn.relu,name='fc_3') #shape [batch_size*(maxTargetNumber*fc3HiddenSize)]

        # fc4
        self.fc_4 = tf.layers.dense(self.fc_3,maxTargetNumber*product_dim,activation=tf.nn.relu,name='fc_4')    #shape [batch_size*(maxTargetNumber*product_dim)]
        self.hid_6 = tf.reshape(self.fc_4,[-1,maxTargetNumber,product_dim],name="hid_6")  #shape [batch_size*maxTargetNumber*product_dim]