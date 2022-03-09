import numpy as np
import mv3d
import mv3d_net
import glob
from sklearn.utils import shuffle
from config import *
# import utils.batch_loading as ub
import argparse
import os
from utils.training_validation_data_splitter import TrainingValDataSplitter
from utils.batch_loading import BatchLoading2 as BatchLoading


def str2bool(v):
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')

    all = '%s,%s,%s,%s' % (mv3d_net.top_view_rpn_name, mv3d_net.imfeature_net_name, mv3d_net.fusion_net_name,
                           mv3d_net.front_feature_name)

    parser.add_argument('-w', '--weights', type=str, nargs='?', default='',
                        help='use pre trained weights example: -w "%s" ' % (all))

    parser.add_argument('-t', '--targets', type=str, nargs='?', default=all,
                        help='train targets example: -w "%s" ' % (all))

    parser.add_argument('-i', '--max_iter', type=int, nargs='?', default=1000,
                        help='max count of train iter')

    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')

    parser.add_argument('-c', '--continue_train', type=str2bool, nargs='?', default=False,
                        help='set continue train flag')

    parser.add_argument('-r', '--raw_dataset', type=str2bool, nargs='?', default=False,
                        help='set using raw or 3d-object-detection')

    args = parser.parse_args()

    print('\n\n{}\n\n'.format(args))
    tag = args.tag
    if tag == 'unknown_tag':
        tag = input('Enter log tag : ')
        print('\nSet log tag :"%s" ok !!\n' % tag)

    max_iter = args.max_iter
    weights = []
    if args.weights != '':
        weights = args.weights.split(',')

    targets = []
    if args.targets != '':
        targets = args.targets.split(',')

    dataset_dir = cfg.PREPROCESSED_DATA_SETS_DIR
    tags_3d = [x for x in range(7481)]
    tags_3d = shuffle(tags_3d, random_state=666)
    training_tags = tags_3d[:3712]
    val_tags = tags_3d[3712:]
    np.save("val_tags.npy", np.array(val_tags))

    with BatchLoading(require_shuffle=True, random_num=np.random.randint(100), tags=training_tags,
                       is_raw=False, data_cate = 'training') as training:
        with BatchLoading(queue_size=1, require_shuffle=True, random_num=666, is_raw=False,
                           data_cate = 'training', tags=val_tags, is_valid=True) as validation:
            train = mv3d.Trainer(train_set=training, validation_set=validation,
                                 pre_trained_weights=weights, train_targets=targets, log_tag=tag,
                                 continue_train=args.continue_train)
            train(max_iter=max_iter)


