import argparse

def getCmdArgs():
    parser = argparse.ArgumentParser(description="Deep GRU net tracker")
    parser.add_argument("--input_data_path",help="input data path",default=None,required=False)
    parser.add_argument("--max_target_number",help="set the largest number of the targetes",default=250)
    parser.add_argument("--gru_hide_size",help="set the gru net hide size",default=16)
    parser.add_argument("--product_dim",help="set the output dim",default=4)
    parser.add_argument("--input_w_size",help="set the input picture weight",default=224)
    parser.add_argument("--input_h_size",help="set the input picture height",default=224)
    parser.add_argument("--det_dim",help="set the det data dim",default=5)
    parser.add_argument("--product_hidden_size",help="set the output hidden lary size",default=32)
    parser.add_argument("--train_path",help="set the train data path",default="train")
    parser.add_argument("--epoch",help="set the trainning epoch",default=10000)
    parser.add_argument("--fc_3_hidden_size",help="set the full conect layer hidden size",default=16)
    parser.add_argument("--batch_size",help="set the batch size",default=1)
    parser.add_argument("--vgg16_npy_path",help="set the vgg16 npy path",default='F:\\work\\workspace\\DeepGRUTracker\\data\\vgg\\model_npy\\vgg16.npy')
    parser.add_argument("--data_path",help="set the input data path",default='F:\\work\\database\\MOT16')
    parser.add_argument("--is_train",help="set train or test",default=True)
    parser.add_argument("--train_folder_name",help="set train folder name",default='train')
    parser.add_argument("--train_image_folder_name",help="set train image folder name",default='img1')
    parser.add_argument("--train_det_folder_name",help="set train det folder name",default='det')
    parser.add_argument("--train_Gt_folder_name",help="set train gt folder name",default='gt')
    parser.add_argument("--train_det_file_name",help="set train det file name",default='det.txt')
    parser.add_argument("--train_Gt_file_name",help="set train gt file name",default='gt.txt')
    parser.add_argument("--test_folder_name",help="set test folder name",default='test')
    parser.add_argument("--test_image_folder_name",help="set test image folder name",default='img1')
    parser.add_argument("--test_det_folder_name",help="set test det folder name",default='det')
    parser.add_argument("--test_det_file_name",help="set test det file name",default='det.txt')
    parser.add_argument("--img_begin_index",help="set img begin index",default=1)
    parser.add_argument("--img_index_bits",help="set img begin index",default=6)
    parser.add_argument("--img_cache_size",help="set img cache size",default=50)
    parser.add_argument("--img_file_type",help="set img file tpye",default='.jpg')
    parser.add_argument("--det_threshold",help="set det keep threshold",default=0.5)
    parser.add_argument("--gru_keep_prob",help="set gru keep prob",default=0.5)
    parser.add_argument("--fc_keep_prob",help="set fc keep prob",default=0.5)
    parser.add_argument("--model_save_path",help="set model save path",default='F:\\work\\workspace\\DeepGRUTracker\\data\\model')
    parser.add_argument("--model_save_epoch",help="set model save epoch",default=3)
    parser.add_argument("--summary_log_save_path",help="set model summary save path",default='F:\\work\\workspace\\DeepGRUTracker\\data\\log')
    return parser.parse_args()