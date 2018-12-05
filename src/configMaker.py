import configparser
import util.getArgs as getArgs

if __name__ == '__main__':
    args = getArgs.getCmdArgs()
    config = configparser.ConfigParser()

    config.add_section('RunConfig')
    config.set('RunConfig','is_train',str(args.is_train))
    config.set('RunConfig','input_data_path',str(args.input_data_path))
    config.set('RunConfig','load_epoch',str(args.load_epoch))
    config.set('RunConfig','det_threshold',str(args.det_threshold))
    config.set('RunConfig','epoch',str(args.epoch))
    config.set('RunConfig','img_begin_index',str(args.img_begin_index))
    config.set('RunConfig','img_index_bits',str(args.img_index_bits))
    config.set('RunConfig','img_cache_size',str(args.img_cache_size))
    config.set('RunConfig','vgg16_npy_path',str(args.vgg16_npy_path))
    config.set('RunConfig','data_path',str(args.data_path))
    config.set('RunConfig','train_path',str(args.train_path))
    config.set('RunConfig','train_folder_name',str(args.train_folder_name))
    config.set('RunConfig','train_image_folder_name',str(args.train_image_folder_name))
    config.set('RunConfig','train_det_folder_name',str(args.train_det_folder_name))
    config.set('RunConfig','train_gt_folder_name',str(args.train_gt_folder_name))
    config.set('RunConfig','train_det_file_name',str(args.train_det_file_name))
    config.set('RunConfig','train_gt_file_name',str(args.train_gt_file_name))
    config.set('RunConfig','sequence_info_file_name',str(args.sequence_info_file_name))
    config.set('RunConfig','test_folder_name',str(args.test_folder_name))
    config.set('RunConfig','test_image_folder_name',str(args.test_image_folder_name))
    config.set('RunConfig','test_det_folder_name',str(args.test_det_folder_name))
    config.set('RunConfig','test_det_file_name',str(args.test_det_file_name))
    config.set('RunConfig','img_file_type',str(args.img_file_type))
    config.set('RunConfig','tracker_result_path',str(args.tracker_result_path))
    config.set('RunConfig','model_save_path',str(args.model_save_path))
    config.set('RunConfig','model_save_epoch',str(args.model_save_epoch))
    config.set('RunConfig','gpu_index',str(args.gpu_index))
    config.set('RunConfig','summary_log_save_path',str(args.summary_log_save_path))

    config.add_section('SuperConfig')
    config.set('SuperConfig','gru_keep_prob',str(args.gru_keep_prob))
    config.set('SuperConfig','nn_keep_prob',str(args.nn_keep_prob))
    config.set('SuperConfig','learn_rate',str(args.learn_rate))

    config.add_section('NetConfig')
    config.set('NetConfig','max_target_number',str(args.max_target_number))
    config.set('NetConfig','gru_hide_size',str(args.gru_hide_size))
    config.set('NetConfig','product_dim',str(args.product_dim))
    config.set('NetConfig','sequence_size',str(args.sequence_size))
    config.set('NetConfig','input_w_size',str(args.input_w_size))
    config.set('NetConfig','input_h_size',str(args.input_h_size))
    config.set('NetConfig','img_resize_w',str(args.img_resize_w))
    config.set('NetConfig','img_resize_h',str(args.img_resize_h))
    config.set('NetConfig','pic_feature_dim',str(args.pic_feature_dim))
    config.set('NetConfig','det_feature_dim',str(args.det_feature_dim))
    config.set('NetConfig','all_feature_dim',str(args.all_feature_dim))
    config.set('NetConfig','det_dim',str(args.det_dim))
    config.set('NetConfig','product_hidden_size',str(args.product_hidden_size))
    config.set('NetConfig','fc_sec_hidden_size',str(args.fc_sec_hidden_size))
    config.set('NetConfig','batch_size',str(args.batch_size))
    config.set('NetConfig','product_hidden_size',str(args.product_hidden_size))
    config.set('NetConfig','test_loop_size',str(args.test_loop_size))
    config.set('NetConfig','gru_larys',str(args.gru_larys))

    config.write(open('F:\\work\\workspace\\DeepGRUTracker\\config\\train.ini','+w'))