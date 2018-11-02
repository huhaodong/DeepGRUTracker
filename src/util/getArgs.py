import argparse

def getCmdArgs():
    parser = argparse.ArgumentParser(description="Deep GRU net tracker")
    parser.add_argument("--input_data_path",help="input data path",default=None,required=False)
    parser.add_argument("--max_target_number",help="set the largest number of the targetes",default=300)
    parser.add_argument("--gru_hide_size",help="set the gru net hide size",default=128)
    parser.add_argument("--product_dim",help="set the output dim",default=4)
    parser.add_argument("--input_w_size",help="set the input picture weight",default=1920)
    parser.add_argument("--input_h_size",help="set the input picture height",default=1080)
    parser.add_argument("--det_dim",help="set the det data dim",default=5)
    parser.add_argument("--product_hidden_size",help="set the output hidden lary size",default=32)
    parser.add_argument("--trainPath",help="set the train data path",default="train")
    parser.add_argument("--epoch",help="set the trainning epoch",default=100)
    parser.add_argument("--fc_3_hidden_size",help="set the full conect layer hidden size",default=64)
    parser.add_argument("--vgg16_npy_path",help="set the vgg16 npy path",default='model/vgg/model_npy/vgg16.npy')
    return parser.parse_args()